cd source/
echo "Treatement ..."
python3 treatement.py
echo "Model ..."
python3 model.py
echo "Prediction ..."
python3 generate_prediction.py
cd ..