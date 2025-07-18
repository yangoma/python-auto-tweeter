"""Add profile_image_url to twitter_accounts

Revision ID: 4f46ef0de30d
Revises: 
Create Date: 2025-07-11 10:07:54.893177

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4f46ef0de30d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('twitter_accounts', sa.Column('profile_image_url', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('twitter_accounts', 'profile_image_url')
    # ### end Alembic commands ###