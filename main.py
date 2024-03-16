from copy import copy
import random

import numpy as np
from Models.PGGAN.utils import GradientPenalty, Progress, exp_mov_avg, hypersphere, printProgressBar
from Models.dcgan import Generator, Discriminator
from Preprocessing.preprocessing import Classifier_Preprocessing, GAN_Preprocessing
from Preprocessing.dataset import Dataset
from Models.classifier import ResNet
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils as vutils
from sklearn.model_selection import KFold
from Models.PGGAN import pggan


def training_classifier(training_data):

    batch_size = 64

    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    train_dataset = Dataset(training_data, transform_train)

    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
    )

    # Define the device (CPU or GPU)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    net = ResNet.ResNet50(10, 1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(train_loader):
            
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0 and i > 0:
                print(f'Loss [{epoch + 1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)

    print('Training Done')

def cross_validation_classifier(training_data):
    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    train_dataset = Dataset(training_data, transform_train)

    # Define the number of folds, batch size and loss function
    k_folds = 5
    batch_size = 64
    loss_function = nn.CrossEntropyLoss()

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Define the device (CPU or GPU)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Initialize the k-fold cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")
        print("----------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        test_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        # Initialize the model and optimizer
        model = ResNet.ResNet50(10, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model on the current fold
        for epoch in range(0, 200):
            # Print epoch
            #print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0
            for i, data in enumerate(train_loader):

                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Generate outputs
                outputs = model(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold + 1, 100.0 * correct / total))
            print('--------------------------------')
            results[fold + 1] = 100.0 * (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    final_sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        final_sum += value
    print(f'Average: {final_sum / len(results.items())} %')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def training_dcgan(training_data):
    # Set random seed for reproducibility
    manualSeed = 42
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 2

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Size of z latent vector (i.e. size of generator input)
    nz = 256

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 550

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    train_dataset = Dataset(training_data, transform_train)

    # Create the dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(ngpu, nz, ngf, nc).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                torchvision.utils.save_image(vutils.make_grid(fake, padding=2, normalize=True), 'Results/dcgan-' + str(iters) + '.png')

            iters += 1

def training_pggan(training_data):
    
    # Batch size during training
    batch_size = 2

    # Number of workers for dataloader
    workers = 2

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Output resolution
    max_res = 3 # for 32x32 output

    # use WeightScale in G and D
    ws = True

    # use BatchNorm in G and D
    bn = True

    # use PixelNorm in G
    pn = True

    # base number of channel for networks
    nch = 64

    # lambda for gradient penalty
    lambdaGP = 10

    # gamma for gradient penalty
    gamma = 1

    # number of epochs to train before changing the progress
    n_iter = 50

    # number of examples images to save
    savenum = 64

    # epsilon drift for discriminator loss
    e_drift = 0.001

    transform = transforms.Compose([
        # resize to 32x32
        transforms.Pad((2, 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = Dataset(training_data, transform)

    # Create the dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Model creation and init
    G = pggan.Generator(max_res, nch, nc, bn=True, ws=True, pn=True).to(device)
    D = pggan.Discriminator(max_res, nch, nc, bn=True, ws=True).to(device)

    if not ws:
    # weights are initialized by WScale layers to normal if WS is used
        G.apply(weights_init)
        D.apply(weights_init)
    Gs = copy.deepcopy(G)

    optimizerG = optim.Adam(G.parameters(), lr=1e-3, betas=(0, 0.99))
    optimizerD = optim.Adam(D.parameters(), lr=1e-3, betas=(0, 0.99))

    GP = GradientPenalty(batch_size, lambdaGP, gamma, device)

    epoch = 0
    global_step = 0
    total = 2

    P = Progress(n_iter, max_res, batch_size)

    P.progress(epoch, 1, total)
    GP.batchSize = P.batchSizer

    lossEpochG = []
    lossEpochD = []
    lossEpochD_W = []

    for i, (images, _) in enumerate(dataloader):
        P.progress(epoch, i + 1, total + 1)
        global_step += 1

        # Build mini-batch
        images = images.to(device)
        images = P.resize(images)

        # ============= Train the discriminator =============#

        # zeroing gradients in D
        D.zero_grad()
        # compute fake images with G
        z = hypersphere(torch.randn(P.batchSize, nch * 32, 1, 1, device))
        with torch.no_grad():
            fake_images = G(z, P.p)

        # compute scores for real images
        D_real = D(images, P.p)
        D_realm = D_real.mean()

        # compute scores for fake images
        D_fake = D(fake_images, P.p)
        D_fakem = D_fake.mean()

        # compute gradient penalty for WGAN-GP as defined in the article
        gradient_penalty = GP(D, images.data, fake_images.data, P.p)

        # prevent D_real from drifting too much from 0
        drift = (D_real ** 2).mean() * e_drift

        # Backprop + Optimize
        d_loss = D_fakem - D_realm
        d_loss_W = d_loss + gradient_penalty + drift
        d_loss_W.backward()
        optimizerD.step()

        lossEpochD.append(d_loss.item())
        lossEpochD_W.append(d_loss_W.item())

        # =============== Train the generator ===============#

        G.zero_grad()

        z = hypersphere(torch.randn(P.batchSize, nch * 32, 1, 1, device))
        fake_images = G(z, P.p)
        # compute scores with new fake images
        G_fake = D(fake_images, P.p)
        G_fakem = G_fake.mean()
        # no need to compute D_real as it does not affect G
        g_loss = -G_fakem

        # Optimize
        g_loss.backward()
        optimizerG.step()

        lossEpochG.append(g_loss.item())

        # update Gs with exponential moving average
        exp_mov_avg(Gs, G, alpha=0.999, global_step=global_step)

        printProgressBar(i + 1, total + 1,
                         length=20,
                         prefix=f'Epoch {epoch} ',
                         suffix=f', d_loss: {d_loss.item():.3f}'
                                f', d_loss_W: {d_loss_W.item():.3f}'
                                f', GP: {gradient_penalty.item():.3f}'
                                f', progress: {P.p:.2f}')


if __name__ == '__main__':

    # Preprocessing - Classifier
    raw_data = Classifier_Preprocessing()
    train, test = raw_data.split_data()

    # Train Classifier
    # cross_validation_classifier(train)
    # training_classifier(train)

    # Pretraining - GAN
    classes = ["aneurysmatic bone cyst", "chondroblastoma", "chondrosarcoma",
                          "enchondroma", "ewing sarcoma", "fibruous dysplasia",
                          "giant cell tumour", "non-ossifying fibroma", "osteochondroma",
                          "osteosarcoma"]
    gan_data = GAN_Preprocessing(classes[8])
    train_gan = gan_data.class_data()

    # Train DCGAN
    # training_dcgan(train_gan)

    # Train PGGAN
    #training_pggan(train_gan)
