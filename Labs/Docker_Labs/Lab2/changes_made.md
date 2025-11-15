# Changes Between Old and New Dockerfile and Training Script

1. Added virtual environment in training stage

2. Installed system dependencies for TensorFlow and images

3. Switched dataset from Iris to CIFAR-10

4. Changed model from MLP to small CNN with residual block

5. Added logging to training.log

6. Added metrics evaluation saved as metrics.json

7. Saved model in both .keras and saved_model formats

8. Added command-line arguments for epochs and batch size

9. Added sample inference check on test data

10. Separated Dockerfile into multistage build

11. Serving stage imports multiple artifacts (my_model.keras, saved_model, metrics.json, training.log)

12. Added healthcheck in serving stage  

13. Added optional GPU support flag TF_FORCE_GPU_ALLOW_GROWTH

14. Upgraded pip and installed requirements inside venv in training stage

15. Updated /train api for CIFAR-10 dataset and also added a new /train api with correct image processing