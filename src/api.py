# Importing Required Libraries
from fastapi import FastAPI
import pickle, uvicorn, os
from pydantic import BaseModel
import pandas as pd
import numpy as np

####################################################################

# Config & Setup
## Variables of environment
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "assets")
ML_COMPONENTS = os.path.join(ASSETSDIRPATH, "ml_components.pkl")

## API Basic Config
app = FastAPI(
    title="Titanic Survival API",
    version="1.0",
    description="Prediction of passengers that survived the Titanic shipwreck",
)

# Loading of assets
with open(ML_COMPONENTS, "rb") as file:
    loaded_items = pickle.load(file)


pipeline_of_my_app = loaded_items["pipeline"]

####################################################################
# API Core
## BaseModel
class ModelInput(BaseModel):
    Pclass: object
    Sex: object
    Age: float
    Fare: float
    Embarked: object
    IsAlone: object


## Utils
def make_prediction(Pclass,Sex,Age, Fare,Embarked, IsAlone):
    df = pd.DataFrame([[Pclass,Sex,Age, Fare,Embarked, IsAlone]], columns= ['Pclass','Sex','Age', 'Fare','Embarked', 'IsAlone'])
    
    # Passing data to pipeline to make prediction
    X = df
    output = pipeline_of_my_app.predict(X).tolist()
    
    # # Labelling Model output
    # if output == 0:
    #     model_output = "Not Survived"
    # else:
    #     model_output = "Survived"

    return output
    
# ENDPOINT
@app.post('/titanic')
async def predict(input: ModelInput):
    """__descr__
    --details---
    """
    output = make_prediction(Pclass = input.Pclass,
                             Sex = input.Sex,
                             Age = input.Age,
                             Fare = input.Fare,
                             Embarked = input.Embarked,
                             IsAlone = input.IsAlone
                             )
    return {'Prediction': output,
            'input': input}

####################################################################
# Execution

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        reload=True,
    )
    