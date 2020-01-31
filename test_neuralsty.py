from styleaug import neural_style
import argparse


main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

imgs_inp = "mycar/data/tub_4_19-12-14/1_cam-image_array_.jpg"
imgs_out = "mycar/data/out2.jpg"
mod = "styleaug/saved_models/candy.pth"
cud = 0
cont_s = 1

main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
eval_arg_parser.add_argument("--content-image", type=str, required=True,
                             help="path to content image you want to stylize")
eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                             help="factor for scaling down the content image")
eval_arg_parser.add_argument("--output-image", type=str, required=True,
                             help="path for saving the output image")
eval_arg_parser.add_argument("--model", type=str, required=True,
                             help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
eval_arg_parser.add_argument("--cuda", type=int, required=True,
                             help="set it to 1 for running on GPU, 0 for CPU")
eval_arg_parser.add_argument("--export_onnx", type=str, default=False,
                             help="export ONNX model to a given file")

args = main_arg_parser.parse_args()

args.cuda = cud
args.content_image = imgs_inp
args.output_image = imgs_out
args.model = mod
args.content_scale = cont_s
args.export_onnx = False
neural_style.stylize(args)
