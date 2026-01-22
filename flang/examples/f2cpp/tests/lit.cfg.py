
import os
import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "f2cpp"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.f90']

config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('%llvmshlibdir', config.llvm_shlib_dir))
config.substitutions.append(('%pluginext', config.llvm_plugin_ext))

llvm_config.add_tool_substitutions(
    [ToolSubst("%flang", command=FindTool("flang"), unresolved="fatal"),
     ToolSubst("%flang_fc1", command=FindTool("flang"), extra_args=["-fc1"], unresolved="fatal"),
     ToolSubst("%clang", command=FindTool("clang"), unresolved="fatal"),
     ToolSubst("%clangxx", command=FindTool("clang++"), unresolved="fatal")],
    config.llvm_tools_dir)

