import time
import numpy as np
import spacy
import torch
from Encoder_Vaswani import Transformer
from matplotlib import pyplot as plt
from preprocessing import fetch_data_vaswani
from torch import nn, optim
from torchinfo import torchinfo


def evaluate(model, labels, data, mask):
    model.eval()
    with torch.no_grad():
        labels = torch.from_numpy(labels).to(device).float()
        data = torch.from_numpy(data).to(device).float()
        mask = torch.from_numpy(mask).to(device)
        final_out = model(data, mask)
        # final_out = torch.softmax(final_out, dim=1)[:, 0]
        plug_in = (np.sign(final_out.reshape(final_out.shape[0]).cpu().detach().numpy() - 0.5) + 1) / 2
        exact_labels = labels[:, 0].cpu().detach().numpy().round(decimals=0)
        diff = np.absolute((plug_in - exact_labels))
        acc = 1 - np.sum(diff) / len(labels)
    return acc


if __name__ == "__main__":
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    label_smoothing = 0.0
    epochs = 2000
    d_model = 300
    N = 2
    d_ff = d_model
    h = 2
    l_max = 6
    dropout = 0.0  # 0.1
    learning_rate = 0.00003  # 0.001  # maximal lr reached is  learning_rate /sqrt(warmup_period)
    # minimal lr: 3.95 * 10^-9, maximal lr: 1.58 * 10^-5
    warmup_period = 400
    eval_ratio = 100
    error = 0.05
    nlp = spacy.load('en_core_web_lg')

    model = Transformer(d_model=d_model,
                        N=N,
                        d_ff=d_ff,
                        h=h,
                        l_max=l_max,
                        dropout=dropout
                        ).to(device)
    print(torchinfo.summary(model))

    criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09)
    lambda1 = lambda epoch: np.minimum((epoch + 1) ** (-0.5), (epoch + 1) * warmup_period ** (-1.5))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    losses = []
    accuracies = []

    labels_train, data_train, mask_train = fetch_data_vaswani(l_max, d_model, h, nlp, label_smoothing,
                                                               easy=True, split='train',
                                                              random_state=None)
    labels_dev, data_dev, mask_dev = fetch_data_vaswani( l_max, d_model, h, nlp, label_smoothing,
                                                         easy=True, split='dev', random_state=None)
    batch_size = len(labels_train)
    # batch_size = 100
    t0 = time.time()
    indices = np.arange(batch_size)
    for epoch in range(epochs):

        np.random.shuffle(indices)

        labels_train_batch = labels_train[indices[:batch_size], :]
        data_train_batch = data_train[indices[: batch_size], :, :]
        mask_train_batch = mask_train[indices[: batch_size], :, :, :]

        labels = torch.from_numpy(labels_train_batch).to(device).float()
        data = torch.from_numpy(data_train_batch).to(device).float()
        mask = torch.from_numpy(mask_train_batch).to(device)

        out = model(data, mask)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()

        losses.append(epoch_loss)

        # if epoch_loss < error:
        ##   break

        lr_scheduler.step()
        if epoch % eval_ratio == 0:
            print("Epoch: " + str(epoch))
            print("Loss on train set: " + str(losses[epoch]))
            dev_acc = evaluate(model, labels_dev, data_dev, mask_dev)
            accuracies.append(dev_acc)
            print("Accuracy on dev set: " + str(dev_acc))
            print("Accuracy on train set: " + str(evaluate(model, labels_train, data_train, mask_train)))
            # if dev_acc > 0.98:
            #    break

    t1 = time.time()
    print("Time for learning: " + str(t1 - t0))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(np.arange(len(losses)), losses)
    ax1.plot(np.arange(len(losses)), 0.25 * np.ones(len(losses)))
    plt.title("Loss on training set")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mean square error')
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    ax2.scatter(np.arange(0, len(losses), eval_ratio), accuracies)
    final_dev_acc = evaluate(model, labels_dev, data_dev, mask_dev)
    print("Accuracy on validation set: " + str(final_dev_acc))
    ax2.scatter([len(losses)], [final_dev_acc])
    plt.title("Accuracy on validation set")
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    plt.show()

    labels_test, data_test, mask_test = fetch_data_vaswani(l_max, d_model, h, nlp, label_smoothing,
                                                           easy=True, split='test',
                                                           random_state=None)
    print("Accuracy on test set: " + str(evaluate(model, labels_test, data_test, mask_test)))
