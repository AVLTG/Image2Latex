# Image2Latex
An app to convert handwritten note/work to latex

Its done by training a tensorflow CNN model on the data from [https://github.com/kirel/detexify-data]([url](https://github.com/kirel/detexify-data)) and in the future IAM Handwritten Forms Dataset. The current CNN uses Conv2d convolutions, might consider making the CNN larger and adding ConvTranspose2d to get rid of MaxPooling depending on how well the model does (I have yet to test it). The issue I am currently working on is the data from detexify is humongous so I need a better way of running through it, will try hardware acceleration but if that is still too long I will look into pruning the data or finding an alternative faster way of converting the data so my model can train on it.
