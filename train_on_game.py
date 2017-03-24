from trainer import trainer

t = trainer(epochs = 10, batch_size = 200, validation_split = 0.1, left_and_right_images = True, left_right_offset = 0.2, root_path = 'C:/Users/muddassir/Downloads/data/data/', test_left_and_right_images = False, tune_model = True)
#t = trainer(epochs = 20, batch_size = 300, validation_split = 0.0)


t.train()

t.set_root_image_path('')

t.test()
