# Feature Identification in RSD using Deep Learning and Neural Network

<p text-align="left">
    <a href="https://github.com/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model/issues" alt="Issues">
        <img src="https://img.shields.io/github/issues/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model" /></a>
    <a href="https://github.com/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model/pulls" alt="Pull Requests">
        <img src="https://img.shields.io/github/issues-pr/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model" /></a>
    <a href="https://github.com/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model/network/members" alt="Forks">
        <img src="https://img.shields.io/github/forks/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model" /></a>
    <a href="https://github.com/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model/stargazers" alt="Stars">
        <img src="https://img.shields.io/github/stars/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model" /></a>
</p>

This project is done as my Internship Project at NRSC (ISRO). In this Readme file I will walk you through the complete details of my project in an easiest way as possible.

## :question: What is ML Vs DL

Basically, **Artificial Intelligence** is a broad category of Computer Science that uses machine intelligence to predict the expected output as human does. Now, coming to **Machine Learning**, It is a subset of Artificial Intelligence that uses data and algorithm to perdict an output, and last **Deep Learning**, is a subset of Machine Learning which is more complicated than of Machine Learning to extract certain features using certain layers called **Neural Network** to predict an expected output.

## :question: What is Semantic Segmentation?

Let's understand this concept with the help of an example. Since, here we are using Remote Sensing Data we can take help with that example.

![Orginal-Image](https://github.com/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model/blob/main/data-set/Tile%201/images/landsat_img_01.jpg?raw=true)

What all things we can see in the above Image?

1. **Roads**
2. **Buildings**
3. **Land**
4. **Trees**
5. **Vehicles**

If we closely look into the image we can extract a lot more features from the image.

Now, you are just thinking where does this **Semantic Segmentation** comes in the picture right?

In traditional Machine Learning models, If we wanted to predict a single object from an image it's possible by training the model using multiple images of the object that we wanted to predict. Now, what if in the case if we wanted to predict multiple objects in a single image itself here comes Semantic Segmentation in the picture.

Here, we will use masks to identify different object in an image and compare it with original image to extract different features of different objects and train the model accoding to that.

![Ground-Truth-Mask](https://github.com/akhil-s-kumar/NRSC-Project-Feature-Identification-in-Remote-Sensing-Data-Using-Neural-Network-Model/blob/main/data-set/Tile%201/masks/landsat_img_01.png?raw=true)

I hope, the above image gave some idea :bulb: 