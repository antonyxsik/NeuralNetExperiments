# Radial Function Experiment 

Small MLPs without sufficiently wide layers often have difficulty approximating some radial/circular functions, such as one of the form $f(x,y) = \sqrt{x^2 + y^2}$. This notebook builds a basic MLP in PyTorch to approximate such a function, and has some cool visuals to demonstrate the performance of the net.


# NeuralNetFromScratch

When learning about machine learning and neural networks, it is a good exercise to write your own neural network using only NumPy (no machine learning libraries like PyTorch, TensorFlow, etc.). 

Files: 
- The train_and_test file is a Jupyter Notebook to actually run the network. 
- The model_NN Python file includes all of the heavy lifting and code for the model. 
- The data_MNIST Python file grabs and wrangles the MNIST data into a format that works for us. 

Here's mine from a while back, recently updated for Python 3.11! 
I used the classic MNIST handwritten numbers data set, and wrote a simple yet modifiable NN that classifies which number was handwritten in the image. 

Feel free to add layers, change the number of epochs, batch sizes, or the learning rate to see if you can get some better performance... I have honestly not spent much time tuning this model at all. It was written more for my understanding and as a tutorial for my classmates who were also interested, rather than for actual performance!
