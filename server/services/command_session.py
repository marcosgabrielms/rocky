"""Estado efêmero da janela de comando, separado da conversa do Koda."""


class CommandSession:
    """Registra quais dispositivos têm uma janela de comando externa ativa.

    O prazo da janela continua sob responsabilidade do corpo físico. Esta classe
    apenas preserva o estado necessário para clientes HTTP sem modo explícito.
    """

    def __init__(self) -> None:
        self._active_devices: set[str] = set()

    def open(self, device_id: str) -> None:
        self._active_devices.add(device_id)

    def is_active(self, device_id: str) -> bool:
        return device_id in self._active_devices

    def close(self, device_id: str) -> None:
        self._active_devices.discard(device_id)
