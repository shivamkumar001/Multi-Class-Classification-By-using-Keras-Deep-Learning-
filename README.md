# Multi-Class-Classification-By-using-Keras-Deep-Learning-

IMPORTANT INSTRUCTION

run code in this order
1. sgsp_data.py    #it will maintain datasets
2. sgsp_model.py   # it will maintain model
3. sgsp_main.py    # it will predict on visualise model

# I personaly design like that for simplicity
# 



In this section I wants to predict the class :
here first i click photos of my friends from image_detect.py code

I take 4 classes ["gopal","pradeep","sannidhya","shivam"]. But you can take multiple classes .
steps:

1: After capture images of my friends . i divided images into training_data and testing data


2: Then resize the images eg:(50*50)


3: convert it into numpy array


4: reshapr numpy image


5: convert training_data into x_train(features),y_train(labels)


6: then convert y_train into categorical (By using OneHotEncoder)


eg: if you have 2 classes 


                   0  1
                   0  0
                   0  0
                   0  0
                   1  0
                   0  0
                   0  0
                   
                  
                  
                  only one row take 1 other remains 0 .thats why its called OneHotEncoder



7: then  take model 


8: make layers .its depends upon you ,how much layers you wants to take.


9: compile the program (optimizer,loss,accuracy)


10: training our model (model.fit . or  model.fitGenerator)


11: predict on test data


12: visualize on graph
