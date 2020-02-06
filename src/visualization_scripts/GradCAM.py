def get_CAM(processed_image, predicted_label, layer_name='block5_conv3'):
    """
    This function is used to generate a heatmap for a sample image prediction.
    
    Args:
        processed_image: any sample image that has been pre-processed using the 
                       `preprocess_input()`method of a keras model
        predicted_label: label predicted by the network for this image
    
    Returns:
        heatmap: heatmap generated over the last convolution layer output 
    """
    
    # this will be the model that would give us the graidents
    model_grad = Model([model.inputs], 
                       [model.get_layer(layer_name).output, model.output])
    
    # Gradient tape gives you everything you need
    with tf.GradientTape() as tape:
        conv_output_values, predictions = model_grad(processed_image)
        loss = predictions[:, predicted_label]
    
    # get the gradients wrt to the chosen layer
    grads_values = tape.gradient(loss, conv_output_values)
    
    # take mean gradient per feature map
    grads_values = K.mean(grads_values, axis=(0,1,2))
    
    # convert to numpy. This is done just for image operations.
    # Check for shapes and you would understand why we performed 
    # the squeeze operation here.
    conv_output_values = np.squeeze(conv_output_values.numpy())
    grads_values = grads_values.numpy()
    
    
    # iterate over each feature map in yout conv output and multiply
    # the gradient values with the conv output values. This gives an 
    # indication of "how important a feature is"
    for i in range(512): # we have 512 features in our last conv layer
        conv_output_values[:,:,i] *= grads_values[i]
    
    # create a heatmap
    heatmap = np.mean(conv_output_values, axis=-1)
    
    # remove negative values
    heatmap = np.maximum(heatmap, 0)
    
    # normalize
    heatmap /= heatmap.max()
    
    del model_grad, conv_output_values, grads_values, loss
   
    return heatmap