package io.github.aqmeraamir;

import org.bukkit.Bukkit;
import org.bukkit.block.Block;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockPlaceEvent;

public class EventListener implements Listener {
    private boolean listening = false;

    @EventHandler
    public void onBlockPlace(BlockPlaceEvent event) {
        if (listening) {
            Block block = event.getBlockPlaced();
            int x = block.getX();
            int y = block.getY();
            int z = block.getZ();

            // Ensure the player is drawing on the canvas
            if (x < 0 || x > 27 || y != 110 || z < 0 || z > 27) {
                Bukkit.broadcastMessage("Invalid drawing area. If you would like to end drawing, run: /recognise-digit stop");
            }
        }
    }

    public void setListening(boolean listening) {
        this.listening = listening;
    }
}
