import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument("--epoch", type=int, default=1000, help="Number of max epochs.")
    parser.add_argument("--data", nargs="?", default="office", help="data name.")
    parser.add_argument(
        "--data_dir", nargs="?", default="./data", help="data path."
    )
    parser.add_argument("--random_seed", type=int, default=2024, help="random seed")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size.")
    parser.add_argument("--layers", type=int, default=1, help="gru_layers")
    parser.add_argument(
        "--hidden_factor",
        type=int,
        default=64,
        help="Number of hidden factors, i.e., embedding size.",
    )
    parser.add_argument(
        "--with_text_emb", type=bool, default=True, help="with additional emb or not"
    )
    parser.add_argument(
        "--text_dim", type=int, default=768, help="Number of item text features dims"
    )
    parser.add_argument(
        "--image_dim", type=int, default=4096, help="Number of item image features dims"
    )
    parser.add_argument(
        "--timesteps", type=int, default=200, help="timesteps for diffusion"
    )
    parser.add_argument(
        "--beta_end", type=float, default=0.02, help="beta end of diffusion"
    )
    parser.add_argument(
        "--beta_start", type=float, default=0.0001, help="beta start of diffusion"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--l2_decay", type=float, default=0, help="l2 loss reg coef.")
    parser.add_argument("--cuda", type=int, default=0, help="cuda device.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="dropout ")
    parser.add_argument("--w", type=float, default=2.0, help="dropout ")
    parser.add_argument("--p", type=float, default=0.1, help="dropout ")
    parser.add_argument(
        "--report_epoch", type=bool, default=True, help="report frequency"
    )
    parser.add_argument(
        "--diffuser_type", type=str, default="mlp1", help="type of diffuser."
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="type of optimizer."
    )
    parser.add_argument("--beta_sche", nargs="?", default="exp", help="")
    parser.add_argument(
        "--descri", type=str, default="", help="description of the work."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.9, help="ratio of item representations."
    )
    parser.add_argument(
        "--bpr_alpha", type=float, default=0.9, help="balance bprloss and diff loss"
    )
    parser.add_argument(
        "--recon_weight", type=float, default=0.1, help="balance bprloss and diff loss"
    )
    parser.add_argument(
        "--reg_alpha", type=float, default=0.01, help="balance bprloss and diff loss"
    )
    parser.add_argument(
        "--prex_weight", type=float, default=0.01, help="2-norm weight"
    )
    parser.add_argument(
        "--ssl_weight", type=float, default=0.00005, help="ssl loss weight"
    )
    parser.add_argument(
        "--n_layers", type=int, default=3, help="number of prediction layers"
    )
    parser.add_argument("--name", type=str, default="test", help="name of the test")
    parser.add_argument("--statesize", type=int, default=10, help="number of position embeddings")
    parser.add_argument("--tstBat", type=int, default=2048, help="Batch size.")
    parser.add_argument("--tstEpoch", type=int, default=10, help="test frequency.")
    parser.add_argument("--flag", type=int, default=2, help="pattern")
    return parser.parse_args()


args = parse_args()
