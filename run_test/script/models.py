from monai.networks.nets import UNETR,UNet
# from monai.networks.nets import SwinUNETR

def Create_UNETR(input_channel, label_nbr,cropSize):

    model = UNETR(
        in_channels=input_channel,
        out_channels=label_nbr,
        img_size=cropSize,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        # feature_size=32,
        # hidden_size=1024,
        # mlp_dim=4096,
        # num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.05,
    )


    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=input_channel,
    #     out_channels=label_nbr,
    #     channels = (16,32,64,128,256),
    #     strides=(2,2,2,2),
    #     dropout=0.05,
    # )

    return model

# def Create_SwinUNETR(input_channel, label_nbr,cropSize):

#     model = SwinUNETR(
#         img_size=cropSize,
#         in_channels=input_channel,
#         out_channels=label_nbr,
#         feature_size=48,
#         # drop_rate=0.0,
#         # attn_drop_rate=0.0,
#         # dropout_path_rate=0.0,
#         use_checkpoint=True,
#     )

#     return model
