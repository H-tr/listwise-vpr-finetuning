

def eval_netvlad():
    # Load the model
    model = load_model('models/netvlad.h5', custom_objects={'NetVLAD': NetVLAD})
    # Load the data
    data = load_data()
    # Evaluate the model
    model.evaluate(data, batch_size=1)