"""
JV-Link COM クライアント

JVDTLab.JVLink を pywin32 経由で操作する
"""
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# 速報系はJVSkip不可
SKIP_DISABLED_SPECS = frozenset({
    "0B31", "0B32", "0B33", "0B34", "0B35", "0B36"
})

# 大きめのバッファ（不足時のエラーを防ぐ）
DEFAULT_BUFFER_SIZE = 256000


class JVLinkClient:
    """JV-Link COMクライアント"""
    
    def __init__(
        self,
        software_id: str = "UNKNOWN",
        service_key: Optional[str] = None,
    ):
        """
        JV-Linkを初期化する
        
        Args:
            software_id: JVInit に渡す Sid（ソフトウェアID）
                ※「利用キー」とは別物。利用キーは JVSetServiceKey で設定する。
            service_key: 利用キー（17桁英数字）をセットしたい場合に指定する。
                - 4-4-4-4-1 のハイフン区切りでもOK（内部でハイフンを除去）
        """
        # pywin32は32bit Pythonでのみ動作
        import win32com.client
        
        self.jvlink = win32com.client.Dispatch("JVDTLab.JVLink")
        self._current_spec: Optional[str] = None
        
        logger.info("Calling JVInit(...)")
        ret = self.jvlink.JVInit(software_id)
        if ret != 0:
            raise RuntimeError(f"JVInit failed: {ret}")

        # ★重要: Sid/キーは秘匿情報になり得るのでログに平文で出さない
        logger.info(f"JVLink initialized sid={self._mask_software_id(software_id)}")

        # 利用キーを指定された場合だけセット（レジストリに保存される）
        if service_key:
            self.set_service_key(service_key)

    @staticmethod
    def _mask_software_id(software_id: str) -> str:
        """ログ出力用にソフトウェアIDをマスクする。"""
        if not software_id:
            return "<empty>"
        s = str(software_id)
        if len(s) <= 8:
            return "*" * len(s)
        return f"{s[:4]}...{s[-4:]}"

    @staticmethod
    def _normalize_service_key(service_key: str) -> str:
        """
        利用キーを JVSetServiceKey 用の17桁英数字に正規化する。

        - 例: 'XXXX-XXXX-XXXX-XXXX-X' -> 'XXXXXXXXXXXXXXXXX' (17 chars)
        """
        if not service_key:
            return ""
        s = str(service_key).strip()
        # ありがちなハイフン類をASCIIに寄せた上で除去
        for h in ("-","‐","‑","‒","–","—","−","－"):
            s = s.replace(h, "-")
        s = s.replace("-", "")
        return s

    def set_service_key(self, service_key: str) -> int:
        """
        利用キー（17桁英数字）をレジストリに保存する。

        Returns:
            0: 成功
           -101: 既に利用キーが登録されている（変更不可）
           -100: 値が不正
           その他: エラー
        """
        sk = self._normalize_service_key(service_key)
        if not sk:
            raise ValueError("service_key is empty")
        if len(sk) != 17:
            raise ValueError(f"service_key length must be 17 after normalization, got {len(sk)}")

        # 仕様書: JVSetServiceKey の引数は 17桁英数字
        ret = self.jvlink.JVSetServiceKey(sk)
        # ★秘密情報は出さない（戻り値だけ）
        logger.info(f"JVSetServiceKey -> {ret}")
        return ret
    
    def open_stored(
        self, 
        data_spec: str, 
        from_time: str, 
        option: int = 1
    ) -> Tuple[int, int, int, str]:
        """
        蓄積系データ取得（JVOpen）
        
        Args:
            data_spec: データ種別ID（"RACE", "0B41"等）
            from_time: 取得開始日時（YYYYMMDDHHmmss形式）
            option: 取得オプション（1=通常, 2=今週, 3=セットアップ）
        
        Returns:
            (ret_code, read_count, download_count, last_file_timestamp)
            - ret_code: 0=成功, 負値=エラー
            - read_count: 読込ファイル数
            - download_count: ダウンロードファイル数
            - last_file_timestamp: 最終ファイルタイムスタンプ
        """
        self._current_spec = data_spec
        result = self.jvlink.JVOpen(data_spec, from_time, option, 0, 0, "")
        # pywin32はtupleで返す
        logger.debug(f"JVOpen({data_spec}, {from_time}) = {result}")
        return result
    
    def open_realtime(self, data_spec: str, race_key: str) -> Any:
        """
        速報系データ取得（JVRTOpen）
        
        Args:
            data_spec: データ種別ID（"0B31", "0B32"等）
            race_key: レースキー（YYYYMMDDJJKKNNRR形式）
        
        Returns:
            COM側の戻り値をそのまま返す（型はCOM実装依存）
            - 正常時: 0 または tuple
            - エラー時: 負値
            
        Note:
            戻り値の型はCOM側の実装に依存するため、
            呼び出し側で柔軟に判定すること
        """
        self._current_spec = data_spec
        result = self.jvlink.JVRTOpen(data_spec, race_key)
        logger.debug(f"JVRTOpen({data_spec}, {race_key}) = {result}")
        return result
    
    def read(self, buffer_size: int = DEFAULT_BUFFER_SIZE) -> Tuple[int, bytes, str]:
        """
        データ読み込み（JVRead）
        
        Args:
            buffer_size: バッファサイズ（デフォルト256000）
        
        Returns:
            (ret_code, data_bytes, filename)
            - ret_code: >0=データあり, 0=EOF, -1=ファイル切替, <-1=エラー
            - data_bytes: 読み込んだデータ（必ずbytes型）
            - filename: ファイル名
        """
        # NOTE:
        #   pywin32経由の戻り値は環境/バージョンにより (ret, data, filename) ではなく
        #   (ret, data, buffsize, filename) のようにバッファサイズが混ざることがある。
        #   ここでは「先頭=ret」「最後のstr=filename」を優先して解釈する。
        result = self.jvlink.JVRead("", buffer_size, "")
        ret_code = result[0] if isinstance(result, tuple) and len(result) > 0 else int(result)

        data = b""
        filename = ""

        if isinstance(result, tuple):
            # data は通常 index=1 に入る（str/bytes）
            if len(result) > 1:
                data = result[1]

            # filename は末尾に入ることが多い（str）
            # 例: (ret, data, 256000, filename)
            for v in reversed(result[1:]):
                if isinstance(v, str):
                    filename = v
                    break
        
        # バッファ不足の可能性をログ（ret_code が必要バイト数を返す実装がある）
        if isinstance(ret_code, int) and ret_code > buffer_size:
            logger.warning(f"Buffer may be insufficient: ret={ret_code}, buf={buffer_size}")
        
        # pywin32がstrで返す場合があるため、必ずbytesに正規化
        if isinstance(data, str):
            data = data.encode("shift_jis", errors="ignore")
        elif not isinstance(data, (bytes, bytearray)):
            data = bytes(data) if data else b""
        
        return ret_code, data, filename

    def file_delete(self, filename: str) -> int:
        """
        ダウンロードしたファイルを削除（JVFiledelete）

        JVRead/JVGets が -402/-403 を返した場合、仕様書では
        「JVFiledelete で該当ファイルを削除し、JVOpen からやり直す」ことが推奨されている。
        """
        if not filename:
            raise ValueError("filename is empty")
        ret = self.jvlink.JVFiledelete(filename)
        logger.info(f"JVFiledelete -> {ret}")
        return ret
    
    def skip(self, data_spec: Optional[str] = None) -> int:
        """
        不要レコードをスキップ（JVSkip）
        
        Args:
            data_spec: 省略時は現在のspecを使用
        
        Returns:
            -1: スキップ不可（速報系）
            その他: JVSkipの戻り値
        
        Note:
            速報系(0B3x)ではJVSkipが失敗するケースがあるため、
            呼び出し側で読み捨てる方が安全
        """
        spec = data_spec or self._current_spec
        if spec in SKIP_DISABLED_SPECS:
            logger.debug(f"JVSkip disabled for {spec}")
            return -1
        return self.jvlink.JVSkip()
    
    def status(self) -> int:
        """
        ダウンロード状況確認（JVStatus）
        
        Returns:
            ダウンロード済みファイル数
        """
        return self.jvlink.JVStatus()
    
    def cancel(self) -> None:
        """ダウンロード中止（JVCancel）"""
        self.jvlink.JVCancel()
        logger.info("JVLink download cancelled")
    
    def close(self) -> int:
        """
        クローズ（JVClose）
        
        Returns:
            0=成功, 負値=エラー
        """
        self._current_spec = None
        ret = self.jvlink.JVClose()
        logger.debug(f"JVClose() = {ret}")
        return ret
    
    def set_ui_properties(self) -> int:
        """
        JVLink設定画面表示（JVSetUIProperties）
        
        Returns:
            0=成功, 負値=エラー
        """
        return self.jvlink.JVSetUIProperties()

