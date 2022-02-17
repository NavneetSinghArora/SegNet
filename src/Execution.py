from Train import Train
from Test import Test

train = Train()
training_loader, validation_loader, testing_loader, visual_loader = train.fetch_data_loaders()
train.visualize_original_data()
train.get_model_parameters()
train.train_model()

test = Test()
test.test_model()