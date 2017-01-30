from trainer import trainer

t = trainer(epochs = 20, batch_size = 100, validation_split = 0.0, left_and_right_images = True, left_right_offset = 0.3, root_path = './')
t.train()
