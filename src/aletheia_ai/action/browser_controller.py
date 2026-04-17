"""Selenium browser automation adapter."""

from __future__ import annotations

import logging

from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException, NoSuchWindowException, WebDriverException
from selenium.webdriver.common.by import By

from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import ActionExecutionError

logger = logging.getLogger(__name__)


class BrowserController:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._driver: webdriver.Remote | None = None

    def open(self, url: str | None = None) -> None:
        target = url or self._config.selenium_base_url
        try:
            driver = self._ensure_driver()
            driver.get(target)
            logger.info("Browser opened", extra={"url": target})
        except WebDriverException as exc:
            if self._is_recoverable_session_error(exc):
                self._reset_driver()
                try:
                    driver = self._ensure_driver()
                    driver.get(target)
                    logger.info("Browser reopened after session recovery", extra={"url": target})
                    return
                except WebDriverException as retry_exc:
                    raise ActionExecutionError(f"Browser open failed after recovery: {retry_exc}") from retry_exc
            raise ActionExecutionError(f"Browser open failed: {exc}") from exc

    def click(self, css_selector: str) -> None:
        try:
            driver = self._ensure_driver()
            element = driver.find_element(By.CSS_SELECTOR, css_selector)
            element.click()
            logger.info("Browser click performed", extra={"selector": css_selector})
        except WebDriverException as exc:
            if self._is_recoverable_session_error(exc):
                self._reset_driver()
                try:
                    driver = self._ensure_driver()
                    element = driver.find_element(By.CSS_SELECTOR, css_selector)
                    element.click()
                    logger.info("Browser click recovered after session reset", extra={"selector": css_selector})
                    return
                except WebDriverException as retry_exc:
                    raise ActionExecutionError(f"Browser click failed after recovery: {retry_exc}") from retry_exc
            raise ActionExecutionError(f"Browser click failed: {exc}") from exc

    def type_text(self, css_selector: str, text: str, clear_first: bool = True) -> None:
        try:
            driver = self._ensure_driver()
            element = driver.find_element(By.CSS_SELECTOR, css_selector)
            if clear_first:
                element.clear()
            element.send_keys(text)
            logger.info("Browser type performed", extra={"selector": css_selector})
        except WebDriverException as exc:
            if self._is_recoverable_session_error(exc):
                self._reset_driver()
                try:
                    driver = self._ensure_driver()
                    element = driver.find_element(By.CSS_SELECTOR, css_selector)
                    if clear_first:
                        element.clear()
                    element.send_keys(text)
                    logger.info("Browser type recovered after session reset", extra={"selector": css_selector})
                    return
                except WebDriverException as retry_exc:
                    raise ActionExecutionError(f"Browser type failed after recovery: {retry_exc}") from retry_exc
            raise ActionExecutionError(f"Browser type failed: {exc}") from exc

    def shutdown(self, close_browser: bool = True) -> None:
        if not close_browser:
            logger.info("Browser left open by request")
            return

        if self._driver is not None:
            self._driver.quit()
            self._driver = None
            logger.info("Browser driver shut down")

    def _ensure_driver(self) -> webdriver.Remote:
        if self._driver is not None:
            return self._driver

        try:
            if self._config.selenium_driver == "edge":
                options = webdriver.EdgeOptions()
                if self._config.browser_headless:
                    options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1440,900")
                if self._config.keep_browser_open:
                    options.add_experimental_option("detach", True)
                self._driver = webdriver.Edge(options=options)

            elif self._config.selenium_driver == "firefox":
                options = webdriver.FirefoxOptions()
                if self._config.browser_headless:
                    options.add_argument("-headless")
                self._driver = webdriver.Firefox(options=options)

            else:
                options = webdriver.ChromeOptions()
                if self._config.browser_headless:
                    options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1440,900")
                if self._config.keep_browser_open:
                    options.add_experimental_option("detach", True)
                self._driver = webdriver.Chrome(options=options)

            return self._driver
        except WebDriverException as exc:
            raise ActionExecutionError(f"Failed to initialize Selenium driver: {exc}") from exc

    def _reset_driver(self) -> None:
        if self._driver is None:
            return
        try:
            self._driver.quit()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._driver = None

    def _is_recoverable_session_error(self, exc: WebDriverException) -> bool:
        if isinstance(exc, (InvalidSessionIdException, NoSuchWindowException)):
            return True
        message = str(exc).lower()
        return "invalid session id" in message or "not connected to devtools" in message
