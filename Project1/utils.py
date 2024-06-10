import tensorflow as tf
import numpy as np
from  model.ResponseModel import ResponseModel
import json

def predicat_output():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    test_images = test_images.reshape(-1, 28*28)
    loaded_model = tf.keras.models.load_model("my_saved_model")  

    image_idx = 12  
    sample_image = np.expand_dims(test_images[image_idx], axis=0)

    predictions = loaded_model.predict(sample_image)
    predicted_label = np.argmax(predictions)

    return (f"True Label: {test_labels[image_idx]}, Predicted Label: {predicted_label}") 


def main():
    result = predicat_output()
    print(result)

# Check if this script is being run as the main program and not imported as a module.
if __name__ == "__main__":
    main()



def prediction_in_percentage(predictions):
    # Convert scientific notation to floating-point numbers
    predictions = predictions.tolist()[0]  # Convert to a regular Python list
    predictions = [float(format(value, ".15f")) for value in predictions]

    # Normalize to percentages
    sum_of_probabilities = sum(predictions)
    percentage_predictions = [value * 100 /
                              sum_of_probabilities for value in predictions]
    

    # Print the percentage predictions
    for digit, percentage in enumerate(percentage_predictions):
        print(f"Digit {digit}: {percentage:.2f}%")
    #print(type(percentage_predictions))

    return percentage_predictions

def custom_encoder(o):
    if isinstance(o, ResponseModel):
        image_data = [
            [int(val) for val in row] for row in o.image
        ]
        return {
            "predicted_digit": int(o.predicted_digit),
            "percentage_prediction": o.percentage_prediction,
            "percentage_predictions": o.percentage_predictions,
            "image": image_data,
        }