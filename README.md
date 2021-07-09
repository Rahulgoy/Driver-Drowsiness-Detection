"# Driver-Drowsiness-Detection" 

These days there area many cases when we have heard that there is an accident due to exhaustion or may I say less sleep.The main of developing this program was to warn the drivers when they are feeling dizziness.

In this project I have mainly used two python libaries opencv and keras.

The main steps involved are:
>>Importing haar cascades to detect face and eyes.
>>Capturing video using cv2 and initialising codec for video writer(i.eXVID)
>>After detecting eyes these images are converted into arrays of shape(24, 24, 1) which is the required format for model predicting(input size of Convolution Neural Network)
>>If eyes are closed for greater than 15 score an alarm is set of warning the driver.



Note: For predicting I have used CNN model.
