from trainer import trainer

t = trainer(epochs = 20, batch_size = 100, validation_split = 0.2, left_and_right_images = True, left_right_offset = 0.3, root_path = 'C:/Users/muddassir/Downloads/data/data/', test_left_and_right_images = False)
#t = trainer(epochs = 20, batch_size = 300, validation_split = 0.0)


t.train()

t.set_root_image_path('')

t.test()
