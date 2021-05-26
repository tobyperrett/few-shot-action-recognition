import torch
import numpy as np
import argparse
import os
import pickle

from torch._C import memory_format
from utils import print_and_log, get_log_files, TestAccuracies, aggregate_accuracy, pt_accuracy, verify_checkpoint_dir
from model import Pretrain_CNN_TSN

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import video_reader
import random 

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.writer = SummaryWriter()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()

        self.video_dataset = video_reader.VideoDataset(self.args, meta_batches=False)
        self.video_loader = torch.utils.data.DataLoader(self.video_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True)
        self.val_accuracies = TestAccuracies([self.args.dataset])

        self.accuracy_fn = aggregate_accuracy
#        self.accuracy_fn = pt_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_epoch = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        if self.args.method == "pt_tsn":
            model = Pretrain_CNN_TSN(self.args)
        
        model = model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(self.args.num_gpus)])
        return model


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", type=str, default="data/ssv2small", help="Path to dataset")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--tasks_per_batch", type=int, default=1, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--test_model_name", "-m", default="checkpoint_best_val.pt", help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=250020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")

        parser.add_argument('--val_iters', nargs='+', type=int, help='iterations to val at. Default is for ssv2 otam split.', default=[30, 60, 70])

        parser.add_argument("--print_freq", type=int, default=1, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=8, help="Num dataloader workers.")
        parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="backbone")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--save_freq", type=int, default=5, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to split the ResNet over")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[30,60])
        parser.add_argument("--method", choices=["pt_tsn"], default="pt_tsn", help="pre_train backbone method to use")
        parser.add_argument("--batch_size", "-b", type=int, default=56, help="Batch size")
        parser.add_argument("--epochs", type=int, default=70, help="Total epochs to train for")
        parser.add_argument("--pretrained_backbone", "-pt", type=str, default=None, help="pretrained backbone path")

        args = parser.parse_args()
        
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if args.backbone == "resnet50":
            args.trans_linear_in_dim = 2048
        else:
            args.trans_linear_in_dim = 512
        
        return args

    def run(self):
        train_accuracies = []
        losses = []


        val_accuraies = [0] * 5
        best_val_accuracy = 0


        
        for epoch in range(self.start_epoch, self.args.epochs):
            iteration = 0
            for task_dict in self.video_loader:

                iteration += 1
                torch.set_grad_enabled(True)

                task_loss, task_accuracy = self.train_task(task_dict)
                train_accuracies.append(task_accuracy)
                losses.append(task_loss)
                
                # optimize
                if ((iteration + 1) % self.args.tasks_per_batch == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # print training stats
            if (epoch + 1) % self.args.print_freq == 0:
                print_and_log(self.logfile,'Epoch {}, Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                            .format(epoch+1, torch.Tensor(losses).mean().item(),
                                    torch.Tensor(train_accuracies).mean().item()))
                train_accuracies = []
                losses = []

            self.scheduler.step()

            if ((epoch + 1) % self.args.save_freq == 0):
                self.save_checkpoint(epoch + 1)
            
            # validate
            if ((epoch + 1) in self.args.val_iters):
                accuracy_dict = self.evaluate("val")
                iter_acc = accuracy_dict[self.args.dataset]["accuracy"]
                val_accuraies.append(iter_acc)
                self.val_accuracies.print(self.logfile, accuracy_dict, mode="val")

                # save checkpoint if best validation score
                if iter_acc > best_val_accuracy:
                   best_val_accuracy = iter_acc
                   self.save_checkpoint(epoch + 1, "checkpoint_best_val.pt")

                # val on test
                accuracy_dict = self.evaluate("test")
                self.val_accuracies.print(self.logfile, accuracy_dict, mode="test")

                # # get out if best accuracy was two validations ago
                # if val_accuraies[-1] < val_accuraies[-3]:
                #     break

        # save the final model
        self.save_checkpoint(epoch + 1, "checkpoint_final.pt")


        # evaluate best validation model if it exists, otherwise evaluate the final model.
        try:
            self.load_checkpoint("checkpoint_best_val.pt")
        except:
            self.load_checkpoint("checkpoint_final.pt")

        accuracy_dict = self.evaluate("test")
        self.val_accuracies.print(self.logfile, accuracy_dict, mode="test")

        self.logfile.close()

    def train_task(self, task_dict):
        """
        For one task, runs forward, calculates the loss and accuract and backprops
        """
        task_dict = self.prepare_task(task_dict)
        model_dict = self.model(task_dict['images'])
        target_logits = model_dict['logits']

        task_loss = self.model.module.loss(task_dict, model_dict)
        task_accuracy = self.accuracy_fn(target_logits, task_dict['target_labels'])

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def evaluate(self, mode="val"):
        self.model.eval()
        with torch.no_grad():

            self.video_loader.dataset.split = mode

            accuracy_dict ={}
            accuracies = []
            iteration = 0
            item = self.args.dataset
            for task_dict in self.video_loader:

                iteration += 1

                task_dict = self.prepare_task(task_dict)
                model_dict = self.model(task_dict['images'])
                target_logits = model_dict['logits']
                accuracy = self.accuracy_fn(target_logits, task_dict['target_labels'])
                accuracies.append(accuracy.item())
                del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            # 95% confidence interval
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
            self.video_loader.dataset.split = "train"
        self.model.train()
        
        return accuracy_dict


    def prepare_task(self, task_dict):
        for k in task_dict.keys():
            task_dict[k] = task_dict[k].to(self.device)
        return task_dict

    def save_checkpoint(self, epoch, name="checkpoint.pt"):
        d = {'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'backbone': self.model.module.backbone.state_dict()}
        
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(epoch)))   
        torch.save(d, os.path.join(self.checkpoint_dir, name))

    def load_checkpoint(self, name="checkpoint.pt"):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, name))
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

def main():
    learner = Learner()
    learner.run()

if __name__ == "__main__":
    main()
