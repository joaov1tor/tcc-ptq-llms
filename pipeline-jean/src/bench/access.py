"""Check Hub access to each model before downloading anything (Llama 3 is gated)."""

from __future__ import annotations

from dataclasses import dataclass

HINT_TOKEN = (
    "token ausente ou inválido: rode `hf auth login` ou exporte HF_TOKEN "
    "(passo a passo no README, seção 'Acesso aos modelos')"
)
HINT_GATED = (
    "acesso ainda não aprovado: peça em https://huggingface.co/{model_id} "
    "e acompanhe em https://huggingface.co/settings/gated-repos"
)


@dataclass
class AccessResult:
    model_id: str
    ok: bool
    sha: str | None = None
    reason: str | None = None


def has_token() -> bool:
    from huggingface_hub import get_token

    return get_token() is not None


def check_model_access(model_id: str, revision: str | None = None) -> AccessResult:
    from huggingface_hub import auth_check, model_info
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        auth_check(model_id)
        sha = model_info(model_id, revision=revision).sha
    except GatedRepoError:
        # The Hub answers "gated" both without a token and without approval.
        if not has_token():
            return AccessResult(model_id, False, reason=f"modelo gated e {HINT_TOKEN}")
        return AccessResult(model_id, False, reason=HINT_GATED.format(model_id=model_id))
    except RepositoryNotFoundError:
        return AccessResult(model_id, False, reason=f"repo não encontrado ou {HINT_TOKEN}")
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            return AccessResult(model_id, False, reason=HINT_TOKEN)
        return AccessResult(model_id, False, reason=f"erro HTTP {status}: {exc}")
    except Exception as exc:  # offline, DNS, proxy
        return AccessResult(model_id, False, reason=f"{type(exc).__name__}: {exc}")
    return AccessResult(model_id, True, sha=sha)


def format_result(result: AccessResult) -> str:
    if result.ok:
        return f"✅ {result.model_id} @ {result.sha}"
    return f"❌ {result.model_id}: {result.reason}"