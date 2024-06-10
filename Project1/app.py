from fastapi import FastAPI, File, UploadFile, Request
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
#from Project1 import utils
import json
import utils
from model.ResponseModel import ResponseModel
from fastapi.responses import JSONResponse


app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="D:\Applications\python_env\ML\Project1\static"), name="static")

templates = Jinja2Templates(directory="D:\Applications\python_env\ML\Project1\static")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

model = tf.keras.models.load_model('D:\Applications\python_env\ML\Project1\my_saved_model - working')

@app.post("/predict1/")
async def predict_digit(file: UploadFile):
    image = Image.open(file.file)
    image = image.resize((28, 28))
    image = image.convert('L')
    image_array = np.array(image)
    image_array=image_array/255
    image_array = image_array.astype(int)

    image_array_return = [[0] * 28 for _ in range(28)]
    i=0
    j=0
    for row in image_array:
        for pixel in row:
            print(pixel, end=" ")
            image_array_return[i][j]=pixel
            j=j+1
        print() 
        j=0
        i=i+1
    #print(type(image_array))  
    #print(image_array_return)
    image_vector = image_array.reshape(1, 784) 
    '''for row in image_vector:
        for pixel in row:
            print(pixel, end=" ")
        print() '''
    prediction = model.predict(image_vector)

    predicted_digit = np.argmax(prediction)
    print(predicted_digit)
    percentage_prediction=np.max(prediction)
    print(percentage_prediction)
    percentage_predictions=utils.prediction_in_percentage(prediction)
    
    response_model = ResponseModel()
    response_model.predicted_digit=int(predicted_digit)
    response_model.percentage_prediction=float(percentage_prediction*100)
    response_model.percentage_predictions=percentage_predictions
    response_model.image=image_array_return
    #print(utils.custom_encoder(response_model))

    return JSONResponse(content=utils.custom_encoder(response_model))



from pydantic import BaseModel

class ImageData(BaseModel):
    image_base64: str

@app.post("/predict/")
async def predict_digit(image_data: ImageData):
    # Convert the base64 image data to a NumPy array
    import base64
    import io
    from PIL import Image
    
    try:
        # Decode the base64 image data
        image_binary = base64.b64decode(image_data.image_base64)
        image = Image.open(io.BytesIO(image_binary))
        image = image.resize((28, 28))
        image = image.convert('L')
        image_array = np.array(image)
        image_array=image_array/255.0
        image_array = image_array.astype(int)

        image_array_return = [[0] * 28 for _ in range(28)]
        i=0
        j=0
        for row in image_array:
            for pixel in row:
               print(pixel, end=" ")
               image_array_return[i][j]=pixel
               j=j+1
            print() 
            j=0
            i=i+1
            
        image_array = image_array.reshape(1, 28, 28, 1)  # for conv2D

        #image_vector = image_array.reshape(1, 784) 
       
        '''for row in image_vector:
        for pixel in row:
            print(pixel, end=" ")
        print() '''
        prediction = model.predict(image_array)

        predicted_digit = np.argmax(prediction)
        print(predicted_digit)
        percentage_prediction=np.max(prediction)
        print(percentage_prediction)
        percentage_predictions=utils.prediction_in_percentage(prediction)
        # Specify the path where you want to save the PNG image
        #image_path = "E:\Downloads\path_to_save_image.png"

        # Save the binary image to a local file in PNG format
        #with open(image_path, "wb") as image_file:
        #    image_file.write(image_binary)

        # Return a response
        response_model = ResponseModel()
        response_model.predicted_digit=int(predicted_digit)
        response_model.percentage_prediction=float(percentage_prediction*100)
        response_model.percentage_predictions=percentage_predictions
        response_model.image=image_array_return
    #print(utils.custom_encoder(response_model))

        return JSONResponse(content=utils.custom_encoder(response_model))
    except Exception as e:
        print(e)
        return {"error": str(e)}
    
