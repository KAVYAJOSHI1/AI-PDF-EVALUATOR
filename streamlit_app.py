"""Safe root-level Streamlit entrypoint.

Keeping this launcher outside the ``app`` package prevents Streamlit's script
module from shadowing the package named ``app`` during imports.
"""

from app.frontend.app import main


if __name__ == "__main__":
    main()
