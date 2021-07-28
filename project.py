import torch
from train_model_class import Model

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

class Args:
  def __init__(self, mode: str='rgb', exp_name: str="hmdb-test", batch_size: int=32, length: int=32, 
               learnable: str="[0,1,1,1,1]", niter: int=20, system: str="hmdb", model: str="3d"):
    self.mode = mode
    self.exp_name = exp_name
    self.batch_size = batch_size
    self.length = length
    self.learnable = learnable
    self.niter = niter
    self.system = system
    self.model = model
    
 
# args = Args(mode="rgb", exp_name="hmdb-test", batch_size=32, length=32, learnable="[0,1,1,1,1]", niter=20, system="hmdb", model="3d")
args = Args()
model = Model(device)

model.train(args)
