"""Tests for the config module. Delete any __econpizza__ or __jax_cache__ folders you might have in the current folder before running"""
import pytest
import jax
from unittest.mock import patch
import shutil
import os
import sys
# autopep8: off
sys.path.insert(0, os.path.abspath("."))
import econpizza as ep
from econpizza.config import EconPizzaConfig
# autopep8: on

@pytest.fixture(scope="function", autouse=True)
def ep_config_reset():
    ep.config = EconPizzaConfig()

@pytest.fixture(scope="function", autouse=True)
def os_getcwd_create():
      test_cache_folder = os.path.abspath("config_working_dir")

      if not os.path.exists(test_cache_folder):
          os.makedirs(test_cache_folder)

      with patch("os.getcwd", return_value=test_cache_folder):
        yield
      
      if os.path.exists(test_cache_folder):
        shutil.rmtree(test_cache_folder)

def test_config_default_values():
  assert ep.config["enable_persistent_cache"] == False
  assert ep.config.econpizza_cache_folder == "__econpizza_cache__"

def test_jax_persistent_cache_config_default_values():
  assert ep.config["enable_jax_persistent_cache"] == False
  assert ep.config.jax_cache_folder == "__jax_cache__"

def test_config_jax_default_values():
   assert jax.config.values["jax_compilation_cache_dir"] is None
   assert jax.config.values["jax_persistent_cache_min_entry_size_bytes"] == .0
   assert jax.config.values["jax_persistent_cache_min_compile_time_secs"] == 1.0

@patch("os.makedirs")
# @pytest.mark.skip(reason="Skipping until enable_persistent_cache gets exposed for end users")
def test_config_enable_persistent_cache(mock_makedirs):
  ep.config["enable_persistent_cache"] = True
  mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "__econpizza_cache__"), exist_ok=True)

@patch("os.makedirs")
@patch("jax.config.update")
def test_config_enable_jax_persistent_cache(mock_jax_update, mock_makedirs):
  ep.config["enable_jax_persistent_cache"] = True
  mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "__jax_cache__"), exist_ok=True)

  mock_jax_update.assert_any_call("jax_compilation_cache_dir", os.path.join(os.getcwd(), "__jax_cache__"))
  mock_jax_update.assert_any_call("jax_persistent_cache_min_compile_time_secs", 0)

@patch("os.makedirs")
# @pytest.mark.skip(reason="Skipping until enable_persistent_cache gets exposed for end users")
def test_config_set_econpizza_folder(mock_makedirs):
  ep.config.econpizza_cache_folder = "test1"
  ep.config["enable_persistent_cache"] = True

  mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "test1"), exist_ok=True)

@patch("os.makedirs")
@patch("jax.config.update")
def test_config_set_jax_folder(mock_jax_update, mock_makedirs):
  ep.config.jax_cache_folder = "test1"
  ep.config["enable_jax_persistent_cache"] = True
  mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "test1"), exist_ok=True)
  mock_jax_update.assert_any_call("jax_compilation_cache_dir", os.path.join(os.getcwd(), "test1"))

@patch("jax.config.update")
def test_config_jax_folder_set_from_outside(mock_jax_update):
    mock_jax_update("jax_compilation_cache_dir", "jax_from_outside")
    ep.config["enable_jax_persistent_cache"] = True
    mock_jax_update.assert_any_call("jax_compilation_cache_dir", "jax_from_outside")

@patch("os.path.exists")
@patch("os.makedirs")
# @pytest.mark.skip(reason="Skipping until enable_persistent_cache gets exposed for end users")
def test_econpizza_cache_folder_not_created_second_time(mock_makedirs, mock_exists):
  # Set to first return False when the folder is not created, then True when the folder is created
  mock_exists.side_effect = [False, True]

  # When called for the first time, a cache folder should be created(default is __econpizza_cache__)
  ep.config["enable_persistent_cache"] = True
  mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "__econpizza_cache__"), exist_ok=True)
  # Now reset the mock so that the calls are 0 again.
  mock_makedirs.reset_mock()
  # The second time we should not create a folder
  ep.config["enable_persistent_cache"] = True
  mock_makedirs.assert_not_called()

@patch("os.path.exists")
@patch("os.makedirs")
@patch("jax.config.update")
def test_jax_cache_folder_not_created_second_time(mock_jax_update, mock_makedirs, mock_exists):
   # Set to first return False when the folder is not created, then True when the folder is created
   mock_exists.side_effect = [False, True]
   
   # When called for the first time, a cache folder should be created(default is __jax_cache__)
   ep.config["enable_jax_persistent_cache"] = True
   mock_makedirs.assert_any_call(os.path.join(os.getcwd(), "__jax_cache__"), exist_ok=True)
   assert mock_jax_update.call_count == 2
   # Now reset the mock so that the calls are 0 again.
   mock_makedirs.reset_mock()
   mock_jax_update.reset_mock()
   # The second time we should not create a folder
   ep.config["enable_jax_persistent_cache"] = True
   mock_makedirs.assert_not_called()
   assert mock_jax_update.call_count == 0

# @pytest.mark.skip(reason="Skipping until enable_persistent_cache gets exposed for end users")
def test_config_enable_persistent_cache_called_after_model_load():
    _ = ep.load(ep.examples.dsge)

    assert os.path.exists(ep.config.econpizza_cache_folder) == False
    ep.config["enable_persistent_cache"] = True
    assert os.path.exists(ep.config.econpizza_cache_folder) == True

def test_config_enable_jax_persistent_cache_called_after_model_load():
    _ = ep.load(ep.examples.dsge)

    assert os.path.exists(ep.config.jax_cache_folder) == False
    ep.config["enable_jax_persistent_cache"] = True
    assert os.path.exists(ep.config.jax_cache_folder) == True
