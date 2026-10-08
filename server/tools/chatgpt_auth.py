"""Preparação da sessão local; executar manualmente ou pelo launcher Windows."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.auth import AuthError, ChatGPTSession


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Restaurar, renovar ou autorizar a sessão ChatGPT do Koda.")
    parser.add_argument("--profile-from", type=Path, help="Importar uma vez somente o perfil público da POC.")
    arguments = parser.parse_args(argv)
    try:
        session = ChatGPTSession()
        if arguments.profile_from is not None:
            session.import_public_profile(arguments.profile_from)
        session.ensure()
    except AuthError as error:
        print("[KODA] Sessao ChatGPT indisponivel: " + str(error), flush=True)
        return 1
    except KeyboardInterrupt:
        print("[KODA] Autenticacao cancelada.", flush=True)
        return 1
    except Exception:
        # Do not format third-party exceptions: they may include OAuth response bodies.
        print("[KODA] Falha ao preparar sessao ChatGPT.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
