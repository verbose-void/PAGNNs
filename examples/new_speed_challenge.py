import torch
from torchvision import transforms
import tqdm

from examples.speed_challenge_utils import SpeedChallenge, get_args, get_loaders, PConv


@torch.no_grad()
def evaluate(dataloader, save_results_file=None):
    total_loss = 0
    model.eval()
    output_predictions = []
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), disable=not args.tqdm):
            if type(data) == list:
                X, T = data[0].to(device), data[1].to(device)
                # Y = random_guess()
                Y = model(X, force_no_reset=False)
                assert Y.shape == T.shape
                loss = criterion(Y, T)
                total_loss += loss
                if not args.tqdm and (i+1) % args.print_freq  == 0:
                    print('[EVAL %i/%i] loss: %f' % (i, len(dataloader), total_loss.item() / (i + 1)))

            else:
                X = data[0].unsqueeze(0).to(device)
                # print(X)
                Y = model(X)
                output_predictions.extend([(str(pred) + '\n') for pred in Y[0].tolist()])

    if len(output_predictions) > 0 and save_results_file is not None:
        F = open(save_results_file, 'w')
        F.writelines(output_predictions)
        F.close()
        return

    avg_loss = (total_loss / len(dataloader))
    # print('[EVAL Epoch %i] average loss' % epoch, avg_loss)
    return avg_loss


if __name__ == '__main__':
    args = get_args()
    train_loader, eval_loader, test_loader = get_loaders(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = PConv().to(device)
    criterion = torch.nn.functional.mse_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    print(model)

    for epoch in range(args.epochs):
        # train
        model.train()
        total_loss = 0
        t = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=not args.tqdm)
        for i, (X, T) in t:
            X, T = X.to(device), T.to(device)

            optimizer.zero_grad()
            Y = model(X)

            loss = criterion(Y, T)
            loss.backward()

            optimizer.step()

            t.set_description('[TRAIN] loss: %f' % (total_loss/(i + 1)))

            if not args.tqdm and (i+1) % args.print_freq  == 0:
                print('[TRAIN %i/%i] loss: %f' % (i, len(train_loader), total_loss / (i + 1)))

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print('[epoch %i] train loss:' % epoch, avg_loss)

        # evaluate
        eval_loss = evaluate(eval_loader)
        print('[epoch %i] eval loss:' % epoch, eval_loss)

        # save test output
        evaluate(test_loader, save_results_file='test_output_epoch%i.txt' % epoch)


