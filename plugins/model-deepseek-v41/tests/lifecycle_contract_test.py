"""Compile-time Cancel contracts using the production method body; no runtime/GPU.

Only Status and BeginRetirement side effects are modeled. This is deliberately
not a substitute for process/cgroup/NCCL integration fault injection.
"""
import argparse
import pathlib
import re
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--compiler", required=True)
parser.add_argument("--source", type=pathlib.Path, default=root / "generation_session.cpp")
args = parser.parse_args()
source = args.source.read_text(encoding="utf-8")
body = source.split("Status GenerationSession::Cancel(Status reason) {", 1)[1].split(
    "\nResult<GenerationSessionState> GenerationSession::Poll()", 1)[0]
enum = re.search(r"enum class GenerationSessionState \{.*?\};",
                 (root / "generation_session.h").read_text(), re.S).group()
harness = r'''
namespace std { template<class T> constexpr T&& move(T& v) { return static_cast<T&&>(v); } }
struct Status {
  bool success;
  constexpr bool ok() const { return success; }
  static constexpr Status Ok() { return {true}; }
  static constexpr Status InvalidArgument(const char*) { return {false}; }
  static constexpr Status FailedPrecondition(const char*) { return {false}; }
};
ENUM
struct GenerationSession {
  GenerationSessionState state_;
  Status failure_;
  int deadline = 42, reaped = 3, begins = 0;
  constexpr Status BeginRetirement(Status reason) {
    failure_ = reason; state_ = GenerationSessionState::kRetiring;
    deadline = 99; reaped = 0; ++begins; return Status::Ok();
  }
  constexpr Status Cancel(Status reason);
};
constexpr Status GenerationSession::Cancel(Status reason) { BODY
using S = GenerationSessionState;
constexpr bool fault_cleanup_preserved() {
  GenerationSession s{S::kCleaningCgroups, {false}};
  for (int i = 0; i < 8; ++i) if (!s.Cancel({false}).ok()) return false;
  return s.state_ == S::kCleaningCgroups && s.deadline == 42 && s.reaped == 3 && s.begins == 0;
}
constexpr bool normal_cleanup_cancelled_once() {
  GenerationSession s{S::kCleaningCgroups, {true}};
  if (!s.Cancel({false}).ok() || s.state_ != S::kRetiring || s.begins != 1) return false;
  s.deadline = 71; s.reaped = 3;
  if (!s.Cancel({false}).ok() || s.deadline != 71 || s.reaped != 3) return false;
  s.state_ = S::kCleaningCgroups;
  return s.Cancel({false}).ok() && s.state_ == S::kCleaningCgroups && s.begins == 1 && s.deadline == 71;
}
constexpr bool terminal_and_invalid_contracts() {
  constexpr S states[]{S::kRetiring, S::kFailedRetired, S::kFailedUnreconciled};
  for (S state : states) {
    GenerationSession s{state, {false}};
    if (!s.Cancel({false}).ok() || s.state_ != state || s.begins) return false;
  }
  GenerationSession complete{S::kComplete, {true}}, running{S::kRunning, {true}};
  return !complete.Cancel({false}).ok() && !running.Cancel({true}).ok() &&
      running.state_ == S::kRunning && !running.begins;
}
static_assert(fault_cleanup_preserved());
static_assert(normal_cleanup_cancelled_once());
static_assert(terminal_and_invalid_contracts());
'''.replace("ENUM", enum).replace("BODY", body)
with tempfile.TemporaryDirectory(prefix="v41-cancel-") as directory:
    unit = pathlib.Path(directory) / "contract.cpp"
    unit.write_text(harness, encoding="utf-8")
    result = subprocess.run([args.compiler, "-std=c++20", "-fsyntax-only", str(unit)],
                            capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    if result.returncode:
        raise SystemExit(result.returncode)
print("PASS: 3 compile-time lifecycle contracts; no binaries or model execution")
