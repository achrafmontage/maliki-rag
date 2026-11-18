import argparse
import sys

from .config import load_config


def main():
    ap = argparse.ArgumentParser(prog="maliki")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("index")
    a = sub.add_parser("ask"); a.add_argument("question")
    e = sub.add_parser("eval"); e.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "ingest":
        from .ingest import build_chunks
        print(build_chunks(cfg), "مقطع")
    elif args.cmd == "index":
        from .index import build_indexes
        build_indexes(cfg)
    elif args.cmd == "ask":
        from .pipeline import Pipeline
        print(Pipeline(cfg).ask(args.question)["markdown"])
    elif args.cmd == "eval":
        from eval.evaluate import run
        run(cfg, full=args.full)


if __name__ == "__main__":
    sys.exit(main())
