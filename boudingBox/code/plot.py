import matplotlib.pyplot as plt


START = 1000


fo = open('./result1.txt', "r")
lines = fo.readlines()

x = list(range(2500))
y = [float(acc) for acc in lines]

plt.plot(x[START:], y[START:], color='red', linewidth=0.5, label="acc")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title("prediction accuracy on COCO dataset")

# plt.show()
plt.savefig('./result.png', dpi=600)
plt.close('all')
