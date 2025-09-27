import uvicorn
from fastapi import FastAPI, HTTPException

from src.core.assistant import Assistant
from typing import Annotated, Dict, Union

from src.models.api_models import ModelOutput, ModelInput

med_assistant = Assistant()

app = FastAPI(title="X-ray report generator API")

@app.get("/")
async def root()-> Dict[str, str]:
    return {"Status": "App is healthy"}

@app.post("/generate-report/", response_model=ModelOutput)
async def generate_report(usr_input: Annotated[ModelInput, "Generated report from the X-ray image."]) -> Dict[str, str]:
    xray_im = usr_input.image
    ground_truth = usr_input.ground_truth

    report: str = med_assistant(image=xray_im)

    return {"generated_report": report}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)