import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
import matplotlib.pyplot as plt
from src.network import DeepQNetwork, DoubleDQN, DuelingDQN
from src.game import FlappyBird
import pandas as pd
import pickle
from tensorboardX import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")

    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=256, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_iters", type=int, default=500000)
    parser.add_argument("--replay_memory_size", type=int, default=20000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="models")
    parser.add_argument("--model_type", type=str, choices=["dqn", "double_dqn", "dueling_dqn"], default="dqn")

    args = parser.parse_args()
    return args


def train_model(opt, model_type):
    # 确保保存目录存在
    if not os.path.exists(opt.saved_path):
        os.makedirs(opt.saved_path)
        
    # 生成带时间戳的模型文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_{timestamp}_final.pth"
    
    start_time = time.time()
    writer = SummaryWriter(os.path.join(opt.log_path, model_type))
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # Initialize model based on type
    if model_type == "dqn":
        model = DeepQNetwork()
    elif model_type == "double_dqn":
        model = DoubleDQN()
    else:  # dueling_dqn
        model = DuelingDQN()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game = FlappyBird()

    # Initialize metrics
    max_score = 0
    total_score = 0
    game_times = 0
    replay_memory = []
    losses = []
    scores = []
    times = []

    game.reset()
    image, reward, terminal = game.next_frame(0)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    for iter in range(opt.num_iters):
        prediction = model(state)[0]
        epsilon = opt.initial_epsilon * (opt.final_epsilon / opt.initial_epsilon) ** (iter / opt.num_iters)

        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(prediction).item()

        next_image, reward, terminal = game.next_frame(action)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        replay_memory.append((state, action, reward, next_state, terminal))
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[:len(replay_memory) - opt.replay_memory_size]

        batch = random.sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        state_batch = torch.cat(state_batch)
        next_state_batch = torch.cat(next_state_batch)
        action_batch = torch.from_numpy(np.array([[0, 1] if action else [1, 0] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])

        if torch.cuda.is_available():
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_batch = state_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        if model_type == "double_dqn":
            # Double DQN update
            next_actions = torch.argmax(current_prediction_batch, dim=1)
            next_q_values = next_prediction_batch.gather(1, next_actions.unsqueeze(1))
        else:
            next_q_values = torch.max(next_prediction_batch, dim=1)[0].unsqueeze(1)

        y_batch = torch.cat(tuple(reward if terminal else reward + opt.gamma * next_q for reward, terminal, next_q in
                                zip(reward_batch, terminal_batch, next_q_values)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        # Record metrics
        losses.append(loss.item())
        scores.append(game.score)
        times.append(time.time() - start_time)

        if terminal:
            total_score += game.score
            max_score = max(max_score, game.score)
            game.reset()
            state, _, _ = game.next_frame(0)
            state = torch.from_numpy(state)
            if torch.cuda.is_available():
                state = state.cuda()
            state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
            game_times += 1
        else:
            state = next_state

        if iter % 1000 == 0:
            print(f"{model_type} - Iteration: {iter}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.4f}, Score: {game.score}")
            writer.add_scalar('loss', loss.item(), iter)
            writer.add_scalar('epsilon', epsilon, iter)
            writer.add_scalar('score', game.score, iter)
            writer.add_scalar('max_score', max_score, iter)

    # Save model with metadata
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_info': {
            'max_score': max_score,
            'total_score': total_score,
            'game_times': game_times,
            'final_loss': losses[-1] if losses else None,
            'timestamp': timestamp,
            'training_duration': time.time() - start_time
        }
    }
    model_save_path = os.path.join(opt.saved_path, model_filename)
    torch.save(save_dict, model_save_path)
    print(f"{model_type} model saved to {model_save_path}")
    print(f"Training info: Max score: {max_score}, Total games: {game_times}")

    return losses, scores, times


def plot_results(results):
    # Plot training time
    plt.figure(figsize=(10, 6))
    plt.bar(['DQN', 'Double DQN', 'Dueling DQN'], 
            [results['dqn']['time'][-1], 
             results['double_dqn']['time'][-1], 
             results['dueling_dqn']['time'][-1]])
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.savefig('plots/training_time.png')
    plt.close()

    # Plot cumulative scores
    plt.figure(figsize=(10, 6))
    for model_type in ['dqn', 'double_dqn', 'dueling_dqn']:
        # 计算累积得分
        cumulative_scores = np.cumsum(results[model_type]['scores'])
        plt.plot(cumulative_scores, label=model_type)
    plt.title('Cumulative Score Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Score')
    plt.legend()
    plt.grid(True)  # 添加网格便于查看
    plt.savefig('plots/cumulative_scores.png')
    plt.close()

    # Plot losses
    plt.figure(figsize=(10, 6))
    for model_type in ['dqn', 'double_dqn', 'dueling_dqn']:
        plt.plot(results[model_type]['losses'], label=model_type)
    plt.title('Loss Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/losses.png')
    plt.close()


def main():
    opt = get_args()
    results = {}
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Train each model
    for model_type in ['double_dqn', 'dueling_dqn', 'dqn']:
        print(f"\nTraining {model_type}...")
        opt.model_type = model_type
        losses, scores, times = train_model(opt, model_type)
        results[model_type] = {
            'losses': losses,
            'scores': scores,
            'time': times
        }
    
    # Plot results
    plot_results(results)


if __name__ == "__main__":
    main()
