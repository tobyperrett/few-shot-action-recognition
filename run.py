import torch
import numpy as np
import argparse
import os
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, aggregate_accuracy, verify_checkpoint_dir
from model import CNN_TRX, CNN_OTAM, CNN_TSN

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
        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()

        self.video_dataset = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.video_dataset, batch_size=1, num_workers=self.args.num_workers)
        
        self.accuracy_fn = aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        if self.args.method == "trx":
            model = CNN_TRX(self.args)
        elif self.args.method == "otam":
            model = CNN_OTAM(self.args)
        elif self.args.method == "tsn":
            model = CNN_TSN(self.args)

        model = model.to(self.device) 

        if self.args.num_gpus > 1:
            model.distribute_model()
        return model


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", type=str, default="data/ssv2", help="Path to dataset")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=[75000])
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1000, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="backbone")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--method", choices=["trx", "otam", "tsn"], default="trx", help="few-shot method to use")

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
        total_iterations = self.args.training_iterations

        iteration = self.start_iteration
        for task_dict in self.video_loader:
            if iteration >= total_iterations:
                break
            iteration += 1
            torch.set_grad_enabled(True)

            task_loss, task_accuracy = self.train_task(task_dict)
            train_accuracies.append(task_accuracy)
            losses.append(task_loss)

            # optimize
            if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()
            if (iteration + 1) % self.args.print_freq == 0:
                # print training stats
                print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                              .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                      torch.Tensor(train_accuracies).mean().item()))
                train_accuracies = []
                losses = []

            if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                self.save_checkpoint(iteration + 1)


            if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                accuracy_dict = self.test()
                print(accuracy_dict)
                self.test_accuracies.print(self.logfile, accuracy_dict)

        # save the final model
        torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        task_dict = self.prepare_task(task_dict)
        model_dict = self.model(task_dict['support_set'], task_dict['support_labels'], task_dict['target_set'])
        target_logits = model_dict['logits']

        task_loss = self.model.loss(task_dict, model_dict) / self.args.tasks_per_batch
        task_accuracy = self.accuracy_fn(target_logits, task_dict['target_labels'])

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def test(self):
        self.model.eval()
        with torch.no_grad():

                self.video_loader.dataset.split = "test"
                accuracy_dict ={}
                accuracies = []
                iteration = 0
                item = self.args.dataset
                for task_dict in self.video_loader:
                    if iteration >= self.args.num_test_tasks:
                        break
                    iteration += 1

                    task_dict = self.prepare_task(task_dict)
                    model_dict = self.model(task_dict['support_set'], task_dict['support_labels'], task_dict['target_set'])
                    target_logits = model_dict['logits']
                    accuracy = self.accuracy_fn(target_logits, task_dict['target_labels'])
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
                self.video_loader.dataset.split = "train"
        self.model.train()
        
        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        for k in task_dict.keys():
            task_dict[k] = task_dict[k][0].to(self.device)
        return task_dict

    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

def main():
    learner = Learner()
    learner.run()

if __name__ == "__main__":
    main()
