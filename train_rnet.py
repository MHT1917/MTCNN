import nets
import train

if __name__ == '__main__':
    net = nets.RNet()

    trainer = train.Trainer(net, './param/rnet.pt', r"F:\celeba1\24")
    trainer.train()
