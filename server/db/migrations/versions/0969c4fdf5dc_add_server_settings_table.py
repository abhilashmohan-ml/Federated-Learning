"""add_server_settings_table

Revision ID: 0969c4fdf5dc
Revises:
Create Date: 2026-08-14 21:19:59.565178

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0969c4fdf5dc"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the server_settings key-value table."""
    op.create_table(
        "server_settings",
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value", sa.String(length=500), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    """Drop the server_settings table."""
    op.drop_table("server_settings")
