from keras.layers import (
    Dense,
    Conv1D,
)
from einops import rearrange



class LinearEmbed:
    def __init__(
        self,
        d_input,
        d_model,
    ):
        """
        input: (B, C, S)
        output: (B, S // 1, d_model)
        """
        super().__init__()
        self.d_input = d_input
        self.emb = Dense(d_model)

    def forward(self, x):
        b, c, s = x.size()
        # assert c == self.d_input, f"Patchsize expected {self.d_input}
        # channels got {c} channels"
        x = x.view(b, c * s // self.d_input, 1)
        return self.emb(x)
    
    
class Conv1DEmbed:
    def __init__(self, d_input, d_model, kernel_size):
        """
        input: (B, C, S)
        """
      
        self.emb = Conv1D(filters=d_model, kernel_size=kernel_size)

    def forward(self, x):
        return rearrange(self.emb(x), "b c s -> b s c")



EMBEDDING_REGISRY = {"conv1d": Conv1DEmbed, "linear": LinearEmbed}
