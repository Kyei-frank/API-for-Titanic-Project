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
    pred_output = pipeline_of_my_app.predict(X).tolist()
    prob_output = np.max(pipeline_of_my_app.predict_proba(X)).tolist()
    
    if pred_output == [0]:
        explanation = 'Passenger did not Survive'
    elif pred_output == [1]:
        explanation = ' Passenger Survived'
    
    return pred_output, prob_output, explanation
    
# ENDPOINT
@app.post('/titanic')
async def predict(input: ModelInput):
    """__INTRODUCTION:__
     
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
    
    DETAILS:
    
The table below gives a description on the variables required to make predictions.
| Variable      | Definition       | Key   |
| :------------ |:---------------: | -----:|
| pclass        | Ticket Class     | 1st / 2nd / 3rd |
| sex           | sex of passenger | male / female |
| Age           | Age of passenger | Enter age of passenger       |
| Fare          | Passenger fare   | Enter Fare of passenger    |
| Embarked      | Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|
| IsAlone       | Whether passenger has relative onboard or not| No = Passenger has relatives on board(parent/Children/spouses/siblings/), Yes = Passenger is Alone |


    """
    pred_output, prob_output, explanation = make_prediction(
                             Pclass = input.Pclass,
                             Sex = input.Sex,
                             Age = input.Age,
                             Fare = input.Fare,
                             Embarked = input.Embarked,
                             IsAlone = input.IsAlone
                             )
    return {'Predicted Class': pred_output,
            'Confidence Probability':prob_output,
            'prediction Explanation': explanation,
            'Input': input
            }

###################################################################
# Execution

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        reload=True,
    )
    