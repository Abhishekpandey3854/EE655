import matplotlib.pyplot as plt

def show_predictions(model, dataset, num=3):
    for images, targets in dataset.take(1):
        preds = model.predict(images)
        for i in range(num):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(images[i])
            axs[0].set_title("Input")
            axs[1].imshow(targets[i])
            axs[1].set_title("SLIC Target")
            axs[2].imshow(preds[i])
            axs[2].set_title("Model Output")
            for ax in axs:
                ax.axis('off')
            plt.show()
