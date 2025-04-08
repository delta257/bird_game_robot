import argparse
import torch
import numpy as np
from src.network import DeepQNetwork, DoubleDQN, DuelingDQN
from src.game import FlappyBird
import os


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=256, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="models")
    parser.add_argument("--model_type", type=str, choices=["dqn", "double_dqn", "dueling_dqn"], default="dqn")
    parser.add_argument("--model_path", type=str, default=None, help="Direct path to the model file")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # 直接指定模型路径
    # 在这里修改模型路径即可
    model_path = r"C:\Users\50597\Desktop\ai基础\FlappyDQN\FlappyDQN\FlappyDQN\models\model_2300000.pth"  # 修改为你想要测试的模型路径
    
    # 如果通过命令行参数指定了模型路径，则使用命令行参数
    if opt.model_path:
        model_path = opt.model_path

    # 直接加载整个模型
    model = torch.load(model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    game = FlappyBird()
    game.reset()
    image, reward, terminal = game.next_frame(0)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_image, reward, terminal = game.next_frame(action)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        state = next_state

        if terminal:
            game.reset()
            image, reward, terminal = game.next_frame(0)
            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()
            state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]


if __name__ == "__main__":
    opt = get_args()
    test(opt)
