# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL
#
# This software and its source code are the exclusive property of Stockcast.
# Unauthorized copying, reproduction, modification, distribution, or use of
# this software, in whole or in part, via any medium, is strictly prohibited
# without the prior written permission of Stockcast.
#
# This software is provided "as is", without warranty of any kind, express or
# implied. Stockcast shall not be liable for any damages arising from the use
# of this software.
#
# For licensing inquiries, contact: legal@stockcast.com
# =============================================================================

LICENSE = """
STOCKCAST PROPRIETARY SOFTWARE LICENSE

Copyright (c) 2026 Stockcast. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL

This software and its source code are the exclusive property of Stockcast.
Unauthorized copying, reproduction, modification, distribution, or use of
this software, in whole or in part, via any medium, is strictly prohibited
without the prior written permission of Stockcast.

RESTRICTIONS:
  1. You may not copy, reproduce, or duplicate this software.
  2. You may not modify, adapt, or create derivative works.
  3. You may not distribute, sublicense, sell, or transfer this software.
  4. You may not reverse engineer, decompile, or disassemble this software.
  5. You may not use this software for any commercial purpose without
     explicit written authorization from Stockcast.

DISCLAIMER:
  This software is provided "as is", without warranty of any kind, express
  or implied, including but not limited to the warranties of merchantability,
  fitness for a particular purpose, and non-infringement. In no event shall
  Stockcast be liable for any claim, damages, or other liability, whether in
  an action of contract, tort, or otherwise, arising from, out of, or in
  connection with the software or the use or other dealings in the software.

For licensing inquiries, contact: legal@stockcast.com
"""

__title__     = "Stockcast"
__version__   = "1.0.0"
__author__     = "Stockcast"
__copyright__ = "Copyright (c) 2026 Stockcast. All Rights Reserved."
__license__   = "Proprietary"
__contact__   = "legal@stockcast.com"


def get_license():
    """Returns the full license text."""
    return LICENSE


def print_license():
    """Prints the full license text to stdout."""
    print(LICENSE)


if __name__ == "__main__":
    print_license()
