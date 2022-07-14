import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
from typing import List, Union
import warnings


class TensorBoardHook:
    """A class for exporting an intermediate tensor of a batch to TensorBoard.
    """

    def __init__(
        self, 
        summary_writer: SummaryWriter,
        name: str,
        module: torch.nn.Module,
    ):
        super().__init__()
        self.summary_writer = summary_writer
        self.name = name
        self.module = module

        self.global_step = None
        self.forward_image = False
        self.forward_image_kwargs = None
        self.forward_histogram = False
        self.forward_histogram_kwargs = None
        self.forward_handle = None
    
    def __call__(
        self,
        module: torch.nn.Module,
        input:Union[torch.Tensor, List[torch.Tensor]],
        output:Union[torch.Tensor, List[torch.Tensor]]
    ) -> None:
        """The "forward hook" function to be registered to the Module.

        Not intended to be called directly. Only used as a hook by the Module.
        """

        if module is not self.module:
            warnings.warn("TensorBoardHook.__call__(): directly called from an unknown module, ignoring.")
            return
        
        if isinstance(output, torch.Tensor):
            output_ = [output]
        else:
            output_ = output
        for idx, out_tensor in enumerate(output_):
            name = "%s.OUTPUT%d" % (self.name, idx)
            if self.forward_image and out_tensor.ndim == 4 and out_tensor.shape[1] == 3:
                self.summary_writer.add_image(
                    name,
                    torchvision.utils.make_grid(out_tensor),
                    global_step=self.global_step,
                    **(self.forward_image_kwargs or {})
                )
            if self.forward_histogram:
                self.summary_writer.add_histogram(
                    name,
                    out_tensor,
                    global_step=self.global_step,
                    **(self.forward_histogram_kwargs or {})
                )

        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
            self.forward_image_kwargs = None
            self.forward_histogram_kwargs = None
            self.forward_image = False
            self.forward_histogram = False
    
    def register_forward(self, global_step=None, *, image=False, histogram=True, image_kwargs=None, histogram_kwargs=None):
        """Registers this hook to the module to enable forward output.

        This method registers the Hook itself to the Module.

        Parameters
        ----------

        global_step: int
            The global_step parameter to be passed to the SummaryWriter.
        image: bool
            Whether or not enable image exporting.
        histogram: bool
            Whether or not enable histogram exporting.
        image_kwargs: Dict[str]
            The keyword args to be passed to SummaryWriter.add_image()
            such as `walltime`.
        histogram_kwargs: Dict[str]
            The keyword args to be passed to SummaryWriter.add_histogram()
            such as `walltime`.
            
        """
        self.global_step = global_step
        self.forward_image_kwargs = image_kwargs
        self.forward_histogram_kwargs = histogram_kwargs
        self.forward_handle = self.module.register_forward_hook(self)
        self.forward_image = image
        self.forward_histogram = histogram


