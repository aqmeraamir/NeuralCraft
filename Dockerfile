FROM pytorch/pytorch:latest

WORKDIR / 

# installing dependencies
COPY ./requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

ENTRYPOINT ["python", "./src/schematic_model.py"]
