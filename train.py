from model import Model, Config
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from Dataset import TrainDataset, ValidDataset
import torch.nn.functional as F

'''
    Adam algorithm
    initial learning rate is 0.001
    the parameters of Adam, β1 and β2 are 0.9 and 0.999 respectively
    employed early-stopping as a regularization strategy
    Models were trained in mini batches with a batch size of 200, and the maximum
    utterance length is 50
    We padded zeros if the number
    of utterances in a context is less than 10, otherwise
    we kept the last 10 utterances
'''
config = Config()
model = Model(config)
train_root_path = ''
valid_root_path = ''
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 200)')
parser.add_argument('--valid-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for validation (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.9, metavar='LR',
                    help='beta 1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='LR',
                    help='beta 2 (default: 0.999)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    TrainDataset(root=train_root_path, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(
    ValidDataset(root=valid_root_path, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=args.valid_batch_size, shuffle=True, **kwargs)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

'''
    nll_loss:
        input:(N,C) where C = number of classes
        target:(N) where each value is 0≤ target ≤C−1,
'''


# early stop 的问题没有解决
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (utterance, response, label) in enumerate(train_loader):
        utterance, response, label = utterance.to(device), response.to(device), label.to(device)
        optimizer.zero_grad()  # 这里边是很多batch 的数据对 对没错 因为模型输入就是batch输入的
        output = model(utterance, response)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(utterance), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def mean_reciprocal_rank(ranks):
    sum = 0
    for rank in ranks:
        sum += (1 / rank)
    return sum / len(ranks)


def rank(lis, idx):
    '''
    查看在lis中 idx位置的数 从大到小排列为第几个数字
    +1 防止等于0
    '''
    temp_list = list(lis)  # 保证不会对形参产生影响
    temp = temp_list[idx]
    temp_list.sort(reverse=True)
    return temp_list.index(temp) + 1


def count_for_r(lis, idx, num):
    '''
    在lis中寻找前num大的值的索引，判断idx是否在这些索引之中 是则返回1，否则返回0
    '''
    temp_list = list(lis)  # 保证不会对形参产生影响
    temp = []
    for i in range(num):
        x = temp_list.index(max(temp_list))
        temp.append(x)
        temp_list[x] = 0

    if idx in temp:
        return 1
    else:
        return 0


def valid(args, model, device, test_loader):
    # 这里边需要评估很多指标
    # MRR MAP P@1 R10@1 R10@2 R10@5
    model.eval()
    correct_1 = 0
    correct_2 = 0
    correct_5 = 0
    with torch.no_grad():
        ranks = []
        for utterance, responses, correct_index in test_loader:
            correct_index = torch.squeeze(correct_index).item()  # 一个数
            responses = responses.permute(1, 0, 2)  # 10,1,self.max_sentence_len
            probabilities = []
            for response in responses:
                utterance, response = utterance.to(device), response.to(device)
                output = model(utterance, response)  # output :(1,2)
                output = torch.squeeze(output)  # output :(2)
                positive_probability = output[1].item()
                probabilities.append(positive_probability)
            r = rank(probabilities, correct_index)
            correct_2 += count_for_r(probabilities, correct_index, 2)
            correct_5 += count_for_r(probabilities, correct_index, 5)
            ranks.append(r)
            if probabilities.index(max(probabilities)) == correct_index:
                correct_1 += 1
        MRR = mean_reciprocal_rank(ranks)
        MAP = MRR
        P_1 = correct_1 / len(test_loader.dataset)
        R10_1 = P_1
        R10_2 = correct_2 / len(test_loader.dataset)
        R10_5 = correct_5 / len(test_loader.dataset)

    print('\nvalid set: MRR = MAP = {}.P_1 = R10_1 = {}.R10_2 = {}.R10_5 = {}\n'.format(MAP, R10_1, R10_2, R10_5))


for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    valid(args, model, device, valid_loader)
if args.save_model:
    torch.save(model.state_dict(), "../model/model.pt")
