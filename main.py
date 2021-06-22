import os
from absl import app
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm
from easydict import EasyDict
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import simplejson
from datetime import datetime
from utils.data_loader import load_dataset
from utils.consts import Consts, ClassEncoder
from utils.early_stopping import EarlyStopping

from networks.vgg import VGG
from networks.resnet import ResNet18
from networks.resnext import ResNeXt29_2x64d
from networks.googlenet import GoogLeNet
from networks.efficientnet import EfficientNetB0
from networks.mobilenetv2 import MobileNetV2
from networks.shufflenetv2 import ShuffleNetV2

def train(net, optimizer, train_loader, epoch, consts: Consts, grayscale_converter):
    """ Train model """
    net.train()
    train_loss = 0.0
    num_correct, num_total = 0, 0

    for x, y in tqdm(train_loader):
        x, y = x.to(consts.DEVICE), y.to(consts.DEVICE)
        if consts.ADV_TRAIN:
            net.inference_mode()
            # Replace clean example with adversarial example for adversarial training
            # x = projected_gradient_descent(net, x, consts.EPS, 0.01, 40, np.inf)
            x = fast_gradient_method(net, x, consts.EPS, np.inf)
        net.non_inference_mode()
        optimizer.zero_grad()

        output_cls, output_aux = net(x)
        if consts.AUXILIARY_TYPE == "Recon":
            loss = net.loss(output_cls, y, output_aux, x, consts.AUXILIARY_TYPE, consts.AUXILIARY_WEIGHT)
        elif consts.AUXILIARY_TYPE == "Fourier":
            x_gray = grayscale_converter(x)
            x_fourier_complex = torch.fft.fftshift(torch.fft.fft2(x_gray))
            x_fourier_mag = torch.abs(x_fourier_complex)
            x_fourier_ph = torch.angle(x_fourier_complex)
            x_fourier = torch.cat((x_fourier_mag, x_fourier_ph), dim=1)
            loss = net.loss(output_cls, y, output_aux, x_fourier, consts.AUXILIARY_TYPE, consts.AUXILIARY_WEIGHT)
        else:
            loss = net.loss(output_cls, y, None, None, consts.AUXILIARY_TYPE, None)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, y_pred = output_cls.max(1)  # model prediction on clean examples
        num_correct += y_pred.eq(y).sum().item()
        num_total += y.size(0)

    train_acc = num_correct / num_total
    train_loss_norm = train_loss / num_total

    tqdm.write(
        "epoch: {}/{}, train loss: {:.3f}, train acc ({}): {:.2f}%".format(
            epoch, consts.NUM_EPOCHS, train_loss_norm, \
            "adversarial data" if consts.ADV_TRAIN else "clean data", \
            train_acc * 100.0
        )
    )

def validate(net, validation_loader, epoch, consts: Consts, grayscale_converter):
    """ Validate model """
    net.eval()

    # Cannot simply use `with torch.no_grad():` here because adversarial generation requires gradient?
    validation_loss = 0.0
    num_correct, num_total = 0, 0

    for x, y in tqdm(validation_loader):
        x, y = x.to(consts.DEVICE), y.to(consts.DEVICE)
        if consts.ADV_TRAIN:
            net.inference_mode()
            x = fast_gradient_method(net, x, consts.EPS, np.inf)
        net.non_inference_mode()

        output_cls, output_aux = net(x)
        if consts.AUXILIARY_TYPE == "Recon":
            loss = net.loss(output_cls, y, output_aux, x, consts.AUXILIARY_TYPE, consts.AUXILIARY_WEIGHT)
        elif consts.AUXILIARY_TYPE == "Fourier":
            x_gray = grayscale_converter(x)
            x_fourier_complex = torch.fft.fftshift(torch.fft.fft2(x_gray))
            x_fourier_mag = torch.abs(x_fourier_complex)
            x_fourier_ph = torch.angle(x_fourier_complex)
            x_fourier = torch.cat((x_fourier_mag, x_fourier_ph), dim=1)
            loss = net.loss(output_cls, y, output_aux, x_fourier, consts.AUXILIARY_TYPE, consts.AUXILIARY_WEIGHT)
        else:
            loss = net.loss(output_cls, y, None, None, consts.AUXILIARY_TYPE, None)

        validation_loss += loss.item()
        
        _, y_pred = output_cls.max(1)  # model prediction on clean examples
        num_correct += y_pred.eq(y).sum().item()
        num_total += y.size(0)
    
    validation_acc = num_correct / num_total
    validation_loss_norm = validation_loss / num_total

    tqdm.write(
        "epoch: {}/{}, validation loss: {:.3f}, validation acc ({}): {:.2f}%".format(
            epoch, consts.NUM_EPOCHS, validation_loss_norm, \
            "adversarial data" if consts.ADV_TRAIN else "clean data", \
            validation_acc * 100.0
        )
    )

    return validation_acc

def test(net, test_loader, consts: Consts):
    """ Evaluate on clean and adversarial data """
    print("\nEvaluating model on clean and adversarial data...")
    net.eval()
    net.inference_mode()

    accuracies = {'clean': [], 'fgm': [], 'pgd': []}

    for rep in range(consts.TEST_REPS):
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        tqdm.write("repetition: {}/{}".format(rep + 1, consts.TEST_REPS))
        for x, y in tqdm(test_loader):
            x, y = x.to(consts.DEVICE), y.to(consts.DEVICE)
            x_fgm = fast_gradient_method(net, x, consts.EPS, np.inf)
            x_pgd = projected_gradient_descent(net, x, consts.EPS, 0.01, 40, np.inf)

            _, y_pred = net(x).max(1)  # model prediction on clean examples
            _, y_pred_fgm = net(x_fgm).max(1)  # model prediction on FGM adversarial examples
            _, y_pred_pgd = net(x_pgd).max(1)  # model prediction on PGD adversarial examples

            report.nb_test += y.size(0)                          # This should be identical over repetitions
            report.correct += y_pred.eq(y).sum().item()          # This should be identical over repetitions
            report.correct_fgm += y_pred_fgm.eq(y).sum().item()  # This shall change over repetitions
            report.correct_pgd += y_pred_pgd.eq(y).sum().item()  # This shall change over repetitions

        accuracies['clean'] += [report.correct / report.nb_test * 100.0]
        accuracies['fgm'] += [report.correct_fgm / report.nb_test * 100.0]
        accuracies['pgd'] += [report.correct_pgd / report.nb_test * 100.0]
    
    acc_text = []
    acc_text.append("test acc on clean examples: {:.3f}\u00B1{:.3f}%".format(
            np.mean(accuracies['clean']), np.std(accuracies['clean'])
        ))
    acc_text.append("test acc on FGM adversarial examples: {:.3f}\u00B1{:.3f}%".format(
            np.mean(accuracies['fgm']), np.std(accuracies['fgm'])
        ))
    acc_text.append("test acc on PGD adversarial examples: {:.3f}\u00B1{:.3f}%".format(
            np.mean(accuracies['pgd']), np.std(accuracies['pgd'])
        ))
    
    for acc in acc_text:
        print(acc)

    return acc_text

def main(_):
    consts = Consts()

    # 1 ################################ Book keeping jobs and logistics ################################
    torch.manual_seed(consts.RANDOM_SEED)
    
    data = load_dataset(consts.DATASET, consts.BATCH_SIZE, consts.NUM_WORKERS)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    weight_path = './saved/saved_weights/' + consts.DATASET + '_' + consts.MODEL + '_' + str(timestamp) + '_'
    consts_path = './saved/saved_settings/' + consts.DATASET + '_' + consts.MODEL + '_' + str(timestamp) + '_consts.txt'

    os.makedirs('/'.join(os.path.abspath(weight_path).split('/')[:-1]), exist_ok = True)
    os.makedirs('/'.join(os.path.abspath(consts_path).split('/')[:-1]), exist_ok = True)

    with open(consts_path, 'w') as file:
        simplejson.dump(ClassEncoder().encode(consts), file)

    # 2 ################# Instantiate model, loss, and optimizer for training ############################
    if consts.AUXILIARY_TYPE == "Fourier":
        decoded_channel = 2
    else:
        decoded_channel = 3

    if consts.MODEL == "VGG11":
        net = VGG(vgg_name='VGG11', img_dim=data.img_dim, decoded_channel=decoded_channel, num_classes=consts.NUM_CLASSES)
    elif consts.MODEL == "ResNet18":
        net = ResNet18(img_dim=data.img_dim, decoded_channel=decoded_channel, num_classes=consts.NUM_CLASSES)
    elif consts.MODEL == "ResNeXt29_2x64d":
        net = ResNeXt29_2x64d(img_dim=data.img_dim, decoded_channel=decoded_channel, num_classes=consts.NUM_CLASSES)
    elif consts.MODEL == "GoogLeNet":
        net = GoogLeNet(img_dim=data.img_dim, decoded_channel=decoded_channel, num_classes=consts.NUM_CLASSES)
    elif consts.MODEL == "EfficientNetB0":
        net = EfficientNetB0(img_dim=data.img_dim, decoded_channel=decoded_channel, num_classes=consts.NUM_CLASSES)
    elif consts.MODEL == "MobileNetV2":
        net = MobileNetV2(img_dim=data.img_dim, decoded_channel=decoded_channel, num_classes=consts.NUM_CLASSES)
    elif consts.MODEL == "ShuffleNetV2":
        net = ShuffleNetV2(net_size=0.5, img_dim=data.img_dim, decoded_channel=decoded_channel, num_classes=consts.NUM_CLASSES)

    net = net.to(consts.DEVICE)
    if consts.DEVICE == "cuda":
        cudnn.benchmark = True

    if consts.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=consts.LEARNING_RATE, momentum=consts.MOMENTUM, weight_decay=consts.WEIGHT_DECAY)
    elif consts.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=consts.LEARNING_RATE)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, verbose=True, patience=5)

    early_stopping = EarlyStopping(criterion_name='Validation Accuracy', mode='max',
                                   patience=10, verbose=True,
                                   path=weight_path+'best_model.pt')

    # This transform is a grayscale conversion. Only used before Fourier Transform.
    grayscale_converter = transforms.Grayscale()

    # 3 ################################### Train, validate and Test #####################################

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, consts.NUM_EPOCHS + 1):
        train(net, optimizer, data.train, epoch, consts, grayscale_converter)
        validation_acc = validate(net, data.validation, epoch, consts, grayscale_converter)

        # scheduler.step()
        scheduler.step(validation_acc)
        early_stopping(validation_acc, net)
        
        if early_stopping.early_stop:
            print("Early stopping!")
            break

    accuracies = test(net, data.test, consts)
    with open(consts_path, 'a') as file:
        for acc in accuracies:
            file.write('\n' + acc)

if __name__ == "__main__":
    app.run(main)
