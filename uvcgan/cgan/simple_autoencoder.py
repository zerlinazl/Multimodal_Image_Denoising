from uvcgan.torch.select             import select_optimizer, select_loss
from uvcgan.torch.image_masking      import select_masking
from uvcgan.models.generator         import construct_generator

from .model_base import ModelBase
from uvcgan.consts          import USE_META

# Sets up generators and feeds in metadata if USE_META is true

class SimpleAutoencoder(ModelBase):

    def _setup_images(self, _config):
        images = [ 'real', 'reco' ]

        if self.masking is not None:
            images.append('masked')

        for img_name in images:
            self.images[img_name] = None

    def _setup_models(self, config):
        # if const USE_META true, construct a Multimodal ViTUNet generator. else the normal one
        if USE_META:
            self.models.encoder = construct_generator(
                config.generator, config.image_shape, self.device, meta=None
            ) 
        else:
            self.models.encoder = construct_generator(
                config.generator, config.image_shape, self.device
            )

    def _setup_losses(self, config):
        self.losses['loss'] = None
        self.loss_fn = select_loss(config.loss)

        assert config.gradient_penalty is None, \
            "Autoencoder model does not support gradient penalty"

    def _setup_optimizers(self, config):
        self.optimizers.encoder = select_optimizer(
            self.models.encoder.parameters(), config.generator.optimizer
        )

    def __init__(
        self, savedir, config, is_train, device, masking = None
    ):
        # pylint: disable=too-many-arguments
        self.masking = select_masking(masking)
        super().__init__(savedir, config, is_train, device)

        assert config.discriminator is None, \
            "Autoencoder model does not use discriminator"

    def print_things(self, thing, tabs=""):
        # function that I wrote to print out everything in the inputs (nested lists and/or tensors) 
        # for debugging purposes.
        if type(thing) == list:
            print(tabs, "list:" , len(thing))
            for t in thing:
                self.print_things(t, tabs + "\t") 
        else:
            print(tabs , thing.shape)
            return 0

    def set_input(self, inputs):

        if USE_META:
            # self.print_things(inputs)
            self.images.real = inputs[0][0].to(self.device)
            self.meta = inputs[0][1].to(self.device)
        else:
            self.images.real = inputs[0].to(self.device)
        

    def forward(self):
        if self.masking is None:
            input_img = self.images.real
        else:
            self.images.masked = self.masking(self.images.real)
            input_img          = self.images.masked

        if USE_META:
            self.images.reco = self.models.encoder(input_img, meta=self.meta) 
        else:
            self.images.reco = self.models.encoder(input_img) 


    def backward(self):
        loss = self.loss_fn(self.images.reco, self.images.real)
        loss.backward()

        self.losses.loss = loss

    def optimization_step(self):
        self.forward()

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        self.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()

