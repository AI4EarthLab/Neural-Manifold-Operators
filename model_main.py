import torch
from torch import nn
from untils import *
from compute_intrinsic_dimension import *


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, spatial_layers):
        super(Encoder, self).__init__()
        strides = stride_generator(spatial_layers)
        self.enc = nn.Sequential(
            ConvSC(input_channels, latent_dim, stride=strides[0]),
            *[ConvSC(latent_dim, latent_dim, stride=s) for s in strides[1:]]
        )
        
    def forward(self, x):  # x: [B*T, C, H, W]
        skip_connection = self.enc[0](x)
        latent = skip_connection
        for layer in self.enc[1:]:
            latent = layer(latent)
        return latent, skip_connection

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, spatial_layers):
        super(Decoder, self).__init__()
        strides = stride_generator(spatial_layers, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(latent_dim, latent_dim, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * latent_dim, latent_dim, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(latent_dim, output_channels, kernel_size=1)
        
    def forward(self, hidden, skip_connection=None):
        for layer in self.dec[:-1]:
            hidden = layer(hidden)
        output = self.dec[-1](torch.cat([hidden, skip_connection], dim=1))
        output = self.readout(output)
        return output

class TimeEvolutionOperator(nn.Module):
    def __init__(self, input_dim, hidden_dim, temporal_layers, inception_kernels=[3, 5, 7, 11], group_count=8):
        super(TimeEvolutionOperator, self).__init__()

        self.temporal_layers = temporal_layers

        # Encoder layers
        enc_layers = [
            Inception(input_dim, hidden_dim // 2, hidden_dim, incep_ker=inception_kernels, groups=group_count)
        ]
        for _ in range(1, temporal_layers - 1):
            enc_layers.append(
                Inception(hidden_dim, hidden_dim // 2, hidden_dim, incep_ker=inception_kernels, groups=group_count)
            )
        enc_layers.append(
            Inception(hidden_dim, hidden_dim // 2, hidden_dim, incep_ker=inception_kernels, groups=group_count)
        )

        # Decoder layers
        dec_layers = [
            Inception(hidden_dim, hidden_dim // 2, hidden_dim, incep_ker=inception_kernels, groups=group_count)
        ]
        for _ in range(1, temporal_layers - 1):
            dec_layers.append(
                Inception(2 * hidden_dim, hidden_dim // 2, hidden_dim, incep_ker=inception_kernels, groups=group_count)
            )
        dec_layers.append(
            Inception(2 * hidden_dim, hidden_dim // 2, input_dim, incep_ker=inception_kernels, groups=group_count)
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        skips = []
        z = x
        for i in range(self.temporal_layers):
            z = self.enc[i](z)
            if i < self.temporal_layers - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.temporal_layers):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z
        return y

class NeuralManifoldOperator(nn.Module):
    def __init__(self, input_shape, intrinsic_dim=None, spatial_dim=16, temporal_dim=256, spatial_layers=4, temporal_layers=8, inception_kernels=[3, 5, 7, 11], group_count=1):
        super(NeuralManifoldOperator, self).__init__()
        time_steps, channels = input_shape

        self.spatial_dim = spatial_dim  

        self.encoder = Encoder(input_channels=channels, latent_dim=spatial_dim, spatial_layers=spatial_layers)

        if intrinsic_dim is not None:
            
            self.projection = nn.Conv2d(spatial_dim, intrinsic_dim, kernel_size=1)
            self.reconstruction = nn.Conv2d(intrinsic_dim, spatial_dim, kernel_size=1)

            self.time_operator = TimeEvolutionOperator(input_dim=time_steps * intrinsic_dim,
                                                       hidden_dim=temporal_dim,
                                                       temporal_layers=temporal_layers,
                                                       inception_kernels=inception_kernels,
                                                       group_count=group_count)
        else:
            # intrinsic_dim is None ----> Pre-training Stage
            self.projection = None
            self.reconstruction = None
            self.time_operator = None  

        self.decoder = Decoder(latent_dim=spatial_dim, output_channels=channels, spatial_layers=spatial_layers)

    def forward(self, input_tensor):
        batch_size, time_steps, channels, height, width = input_tensor.shape
        reshaped_input = input_tensor.view(batch_size * time_steps, channels, height, width)

        latent_representation, skip_connections = self.encoder(reshaped_input)
        _, latent_channels, latent_height, latent_width = latent_representation.shape

        if self.projection is not None:
            projected_latent = self.projection(latent_representation)  # [B*T, intrinsic_dim, H', W']

            projected_latent = projected_latent.view(batch_size, time_steps * projected_latent.shape[1], latent_height, latent_width)

            evolved_latent = self.time_operator(projected_latent)  # [B, T*intrinsic_dim, H', W']

            evolved_latent = evolved_latent.view(batch_size * time_steps, -1, latent_height, latent_width)  # 形状：[B*T, intrinsic_dim, H', W']

            reconstructed_latent = self.reconstruction(evolved_latent)  # [B*T, latent_dim, H', W']
        else:
            reconstructed_latent = latent_representation

        pred_output = self.decoder(reconstructed_latent, skip_connections)
        final_output = pred_output.view(batch_size, time_steps, channels, height, width)

        return final_output

    def get_latent_embeddings(self, input_tensor):
        batch_size, time_steps, channels, height, width = input_tensor.shape
        reshaped_input = input_tensor.view(batch_size * time_steps, channels, height, width)

        latent_representation, _ = self.encoder(reshaped_input)
        return latent_representation  # [B*T, latent_dim, H', W']

if __name__ == "__main__":
    # 1. Reconstruction process
    input_tensor = torch.rand(1, 10, 1, 64, 64)  # B=1, T=10, C=1, H=64, W=64
    model = NeuralManifoldOperator(input_shape=(10, 1))
    output = model(input_tensor)
    print("Pre-training output shape:", output.shape)  #  [1, 10, 1, 64, 64]

    # Latent Embedding --- > ID
    latent_embeddings = model.get_latent_embeddings(input_tensor)
    print("Latent embeddings shape:", latent_embeddings.shape)  # [B*T, latent_dim, H', W']
    latent_embeddings = latent_embeddings.reshape(10, -1)
    print("Latent embeddings shape:", latent_embeddings.shape)
    latent_embeddings_np = latent_embeddings.detach().cpu().numpy()
    intrinsic_dim = estimate_intrinsic_dimension(latent_embeddings_np, k=6)
    print("Estimated intrinsic dimension:", intrinsic_dim)
    
    # 2. Prediction process
    intrinsic_dim = int(round(intrinsic_dim))
    model_with_id = NeuralManifoldOperator(input_shape=(10, 1), intrinsic_dim=intrinsic_dim)
    output_with_id = model_with_id(input_tensor)
    print("Prediction output shape:", output_with_id.shape)  # [1, 10, 1, 64, 64]
