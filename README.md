# Analysis of space rocks using PyTorch. 

&nbsp; This is a simple program with the objective of recreate image classification using this prestigious library.
<br><br><br>

## Requirements

    pip install -r requirements.txt
<br>

## Train
&nbsp;  To load, train and check how the program works, you need to run the main script file *train.py*.

    python train.py
    
    Epoch 0/25
    ====================================================================================================
    train Loss: 0.6629 Acc: 0.6194
    test Loss: 0.6376 Acc: 0.6065
    .
    .
    .
    Epoch 25/25
    ====================================================================================================
    train Loss: 0.2160 Acc: 0.9548
    test Loss: 0.2092 Acc: 0.9677

    Best val Acc: 0.967742

<br>

# Overview
&nbsp; Use of a neural network to learn the associations between features (curves, edges, and texture) and each rock type.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    
 <br>

&nbsp; Once the program completes its first phase will be ready to show some random examples randomly (numSamples). Now we can see some images of our dataset, we can also access them in the '/Data' folder. 

    images, labels = ml_classifier.randomSamples(testTransforms, numSamples=5)
    toPil = transforms.ToPILImage()
    fig = plt.figure(figsize=(20,20))
    classes = ml_classifier.classNames

    for ii in range(len(images)):
      image = toPil(images[ii])
      sub = fig.add_subplot(1, len(images), ii+1)
      plt.imshow(image)
      plt.pause(0.01)

    plt.title = classes
    plt.show()

<br>

##### matplotlib plots
![image 1](resources/1.png) <br>
![image 2](resources/2.png) <br>
![image 3](resources/3.png) <br>
![image 4](resources/5.png) <br>

## Save our model
    # Save pre-trained model
    torch.save(model, 'model.pth')
![image 5](resources/4.png) 

<br>

If you are intereseted on this kind of topic, you could check the following links: <br>
* [PyTorch - Transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) <br>
* [Microsoft learn](https://docs.microsoft.com/en-us/learn/paths/classify-space-rocks-artificial-intelligence-nasa/)
* [Analyze rocks image - Youtube](https://www.youtube.com/watch?v=XoHR4p8AO9o&feature=youtu.be) 
