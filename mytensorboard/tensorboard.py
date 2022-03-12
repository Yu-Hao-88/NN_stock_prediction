from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
    def __init__(self):
        super().__init__()

    def log_on_tensorboard(self, mode, epoch, value):
        # 將指標記錄到 tensorboard 上
        writer = getattr(self, mode + '_writer')
        writer.add_scalar(f'MSE', value, epoch)

    def init_tensorboard_writers(self, model_name):
        # 初始化 tensorboard 要用到的 writer
        create_time = datetime.now()
        create_time = create_time.strftime("%Y-%m-%d_%H-%M-%S")

        log_dir = (f'./data/tensorboard_log/')

        self.train_writer = SummaryWriter(
            log_dir=(f'{log_dir}'
                     f'{model_name}_train_{create_time}'))
        self.valid_writer = SummaryWriter(
            log_dir=(f'{log_dir}'
                     f'{model_name}_valid_{create_time}'))

    def close_tensorboard_writers(self):
        # 將 tensorboard 用到的 writer 關閉
        self.train_writer.close()
        self.valid_writer.close()
